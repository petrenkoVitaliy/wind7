from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from server.configs.models import ModelConfig
from server.configs.predictions import PredictionsConfig
from server.model_handler.model_handler import ModelHandler
from server.utils import L, tprint


class YoloModelHandler(ModelHandler):
    predictions_config: PredictionsConfig
    model_options: ModelConfig
    model: YOLO
    device: str
    use_half: bool
    model_lock: threading.Lock
    is_reloading: threading.Event

    def __init__(
        self, predictions_config: PredictionsConfig, model_options: ModelConfig
    ) -> None:
        self.predictions_config: PredictionsConfig = predictions_config
        self.model_options: ModelConfig = model_options

        self.model: YOLO = YOLO(str(self.model_options.path), task="segment")

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half: bool = self.device == "cuda"

        self.model_lock: threading.Lock = threading.Lock()
        self.is_reloading: threading.Event = threading.Event()

    def get_predictions(self, frame: np.ndarray) -> dict[str, Any] | None:
        if self.is_reloading.is_set():
            return None

        if not self.model_lock.acquire(blocking=False):
            return None

        try:
            get_results_strategy = (
                self._get_track_results
                if self.predictions_config.task == "track"
                else self._get_predict_results
            )
            return self._parse_results(get_results_strategy(frame))

        finally:
            self.model_lock.release()

    async def reload_model(
        self, predictions_config: PredictionsConfig, model_options: ModelConfig
    ) -> None:
        self.is_reloading.set()

        def _reload_task() -> None:
            try:
                tprint(L.RELOAD_YOLO_LOADING, name=predictions_config.model_name)
                new_model = YOLO(str(model_options.path), task="segment")

                dummy_frame = np.zeros(
                    (model_options.size, model_options.size, 3), dtype=np.uint8
                )
                new_model.predict(
                    dummy_frame,
                    imgsz=model_options.size,
                    device=self.device,
                )

                with self.model_lock:
                    self.model = new_model
                    self.predictions_config = predictions_config
                    self.model_options = model_options
            except Exception as e:
                tprint(L.ERROR_YOLO_RELOAD, err=e, exc_info=True)

        try:
            await asyncio.to_thread(_reload_task)
            tprint(L.RELOAD_YOLO_OK, name=predictions_config.model_name)
        except Exception as e:
            tprint(L.ERROR_YOLO_FAIL, err=e, exc_info=True)
        finally:
            self.is_reloading.clear()

    def _get_track_results(self, frame: np.ndarray) -> Results:
        return self.model.track(
            frame,
            device=self.device,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.45,
            tracker=str(Path(__file__).parent / "trackers" / "custom_tracker.yaml"),
            half=self.use_half,
            persist=True,
            verbose=False,
            retina_masks=self.predictions_config.retina_masks,
        )[0]

    def _get_predict_results(self, frame: np.ndarray) -> Results:
        return self.model.predict(
            frame,
            device=self.device,
            task="segment",
            imgsz=self.model_options.size,
            conf=self.predictions_config.conf,
            iou=0.45,
            half=self.use_half,
            retina_masks=self.predictions_config.retina_masks,
            verbose=False,
        )[0]

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def _parse_results(self, results: Results) -> dict[str, Any]:
        t_start = time.perf_counter()

        if results.boxes is None or len(results.boxes) == 0:
            return {"data": [], "metrics": self._get_metrics(results, 0)}

        boxes = results.boxes
        names = results.names

        coords = self._to_numpy(boxes.xyxyn)
        confs = self._to_numpy(boxes.conf)
        cls = self._to_numpy(boxes.cls).astype(int)
        ids = (
            self._to_numpy(boxes.id).astype(int)
            if boxes.id is not None
            else [-1] * len(boxes)
        )

        predictions: list[dict[str, Any]] = [
            {
                "box": [float(c) for c in coords[i]],
                "label": names[cls[i]],
                "id": int(ids[i]),
                "conf": round(float(confs[i]), 2),
            }
            for i in range(len(coords))
        ]

        if results.masks is not None:
            masks_xyn = results.masks.xyn
            for i, p in enumerate(predictions):
                p["mask"] = masks_xyn[i].tolist()
        else:
            for p in predictions:
                p["mask"] = []

        t_end = time.perf_counter()

        map_duration = t_end - t_start
        metrics = self._get_metrics(results, map_duration)

        return {"data": predictions, "metrics": metrics}

    def _get_metrics(self, results: Results, map_duration: float) -> dict[str, float]:
        speed = results.speed
        return {
            "pre": round(float(speed.get("preprocess", 0.0) or 0.0), 1),
            "inf": round(float(speed.get("inference", 0.0) or 0.0), 1),
            "post": round(float(speed.get("postprocess", 0.0) or 0.0), 1),
            "normalize": round(float(map_duration), 1),
            "total": round(
                float(sum(v or 0.0 for v in speed.values()) + map_duration), 1
            ),
        }
