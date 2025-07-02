from ultralytics import YOLO
import asyncio

from server.model_configs import ModelConfig
from server.predictions_config import PredictionsConfig
from server.utils import tprint


class YoloModelHandler:
    def __init__(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        self.predictions_config = predictions_config
        self.model_options = model_options

        self.model = YOLO(str(self.model_options.path), task='segment')

    def get_predictions(self, frame):
        get_results_strategy = self._get_track_results if self.predictions_config.task == "track" else self._get_predict_results

        return self._parse_results(get_results_strategy(frame), frame)

    async def reload_model(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        self.predictions_config = predictions_config
        self.model_options = model_options

        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: YOLO(str(self.model_options.path), task='segment')
            )
            tprint(
                f"RELOAD::YOLO: successfully: {predictions_config.model_name}")
        except Exception as e:
            tprint(f"ERROR::YOLO Failed to reload model: {e}")

    def _get_track_results(self, frame):
        return self.model.track(
            frame,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.5,
            tracker="botsort.yaml",
            half=True,
            persist=True,
            verbose=False,
        )[0]

    def _get_predict_results(self, frame):
        return self.model.predict(
            frame,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.5,
            half=True,
            verbose=False,
        )[0]

    def _parse_results(self, results, frame):
        if results.boxes is None or len(results.boxes) == 0:
            return {"data": [], "metrics": self._get_metrics(results)}

        h, w = frame.shape[:2]
        predictions = []

        boxes = results.boxes
        is_predict = self.predictions_config.task == "predict"
        names = results.names
        has_masks = results.masks is not None

        for i in range(len(boxes)):
            box = boxes[i]

            t_id = -1 if is_predict or box.id is None else int(box.id[0])

            coords = box.xyxy[0]
            norm_box = [
                float(coords[0] / w), float(coords[1] / h),
                float(coords[2] / w), float(coords[3] / h)
            ]

            norm_mask = []
            if has_masks:
                norm_mask = results.masks.xyn[i].tolist()

            predictions.append({
                "box": norm_box,
                "mask": norm_mask,
                "label": names[int(box.cls[0])],
                "id": t_id,
                "conf": round(float(box.conf[0]), 2)
            })

        return {
            "data": predictions,
            "metrics": self._get_metrics(results)
        }

    def _get_metrics(self, results):
        speed = results.speed
        return {
            "pre": round(float(speed.get('preprocess', 0)), 1),
            "inf": round(float(speed.get('inference', 0)), 1),
            "post": round(float(speed.get('postprocess', 0)), 1),
            "total": round(float(sum(speed.values())), 1)
        }
