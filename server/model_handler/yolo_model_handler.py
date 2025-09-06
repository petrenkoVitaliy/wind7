import time
import threading

from ultralytics import YOLO
import asyncio
import numpy as np

from server.model_configs import ModelConfig
from server.predictions_config import PredictionsConfig
from server.utils import tprint


class YoloModelHandler:
    def __init__(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        self.predictions_config = predictions_config
        self.model_options = model_options

        self.model = YOLO(str(self.model_options.path), task='segment')

        self.model_lock = threading.Lock()
        self.is_reloading = False

    def get_predictions(self, frame):
        if self.is_reloading:
            return None

        if not self.model_lock.acquire(blocking=False):
            return None

        try:
            get_results_strategy = self._get_track_results if self.predictions_config.task == "track" else self._get_predict_results
            return self._parse_results(get_results_strategy(frame), frame)

        finally:
            self.model_lock.release()

    async def reload_model(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        self.is_reloading = True

        def _reload_task():
            try:
                tprint(
                    f"RELOAD: Loading new OpenVINO model for {predictions_config.model_name}...")
                new_model = YOLO(str(model_options.path), task='segment')

                tprint("RELOAD: Warming up OpenVINO...")
                dummy_frame = np.zeros(
                    (model_options.size, model_options.size, 3), dtype=np.uint8)
                new_model.predict(
                    dummy_frame, imgsz=model_options.size, verbose=True, device='cpu')

                with self.model_lock:
                    self.model = new_model
                    self.predictions_config = predictions_config
                    self.model_options = model_options
            except Exception as e:
                tprint(f"ERROR: OpenVINO reload failed: {e}")

        try:
            await asyncio.to_thread(_reload_task)
            tprint(
                f"RELOAD::YOLO: successfully: {predictions_config.model_name}")
        except Exception as e:
            tprint(f"ERROR::YOLO Failed to reload model: {e}")
        finally:
            self.is_reloading = False

    def _get_track_results(self, frame):
        return self.model.track(
            frame,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.45,
            tracker="botsort.yaml",
            half=True,
            persist=True,
            verbose=False,
            retina_masks=self.predictions_config.retina_masks
        )[0]

    def _get_predict_results(self, frame):
        return self.model.predict(
            frame,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.45,
            half=True,
            retina_masks=self.predictions_config.retina_masks,
            verbose=False,
        )[0]

    def _parse_results(self, results, _frame):
        t_start = time.perf_counter()

        if results.boxes is None or len(results.boxes) == 0:
            return {"data": [], "metrics": self._get_metrics(results, 0)}

        boxes = results.boxes
        names = results.names
        has_masks = results.masks is not None

        coords = boxes.xyxyn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(
            int) if boxes.id is not None else [-1]*len(boxes)

        masks_data = results.masks.xyn if has_masks else [[]] * len(boxes)

        predictions = [
            {
                "box": [float(c) for c in coords[i]],
                "mask": masks_data[i].tolist() if has_masks else [],
                "label": names[cls[i]],
                "id": int(ids[i]),
                "conf": round(float(confs[i]), 2)
            }
            for i in range(len(coords))
        ]

        t_end = time.perf_counter()

        map_duration = t_end - t_start
        metrics = self._get_metrics(results, map_duration)

        return {
            "data": predictions,
            "metrics": metrics
        }

    def _get_metrics(self, results, map_duration):
        speed = results.speed
        return {
            "pre": round(float(speed.get('preprocess', 0)), 1),
            "inf": round(float(speed.get('inference', 0)), 1),
            "post": round(float(speed.get('postprocess', 0)), 1),
            "normalize": round(float(map_duration), 1),
            "total": round(float(sum(speed.values()) + map_duration), 1)
        }
