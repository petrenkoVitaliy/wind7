from __future__ import annotations

from typing import Any

import numpy as np

from server.configs.models import ModelConfig, ModelsConfig, ModelType
from server.configs.predictions import PredictionsConfig
from server.model_handler.model_handler import ModelHandler
from server.model_handler.onnx_model_handler.onnx_model_handler import OnnxModelHandler
from server.model_handler.yolo_model_handler.yolo_model_handler import YoloModelHandler
from server.utils.formatter import L, tprint


class ModelAdapter:
    def __init__(self, predictions_config: PredictionsConfig) -> None:
        self.model_type: ModelType | None = None
        self.model_handler: ModelHandler | None = None

        self._update_model_handler(predictions_config)

    def _update_model_handler(
        self, predictions_config: PredictionsConfig
    ) -> tuple[bool, ModelConfig]:
        model_options = ModelsConfig[predictions_config.model_name].value

        if model_options.model_type == self.model_type:
            tprint(L.RELOAD_SAME_MODEL_TYPE, mtype=model_options.model_type)
            return False, model_options

        tprint(L.RELOAD_NEW_MODEL_TYPE, mtype=model_options.model_type)

        self.model_type = model_options.model_type

        if self.model_type == ModelType.YOLO:
            self.model_handler = YoloModelHandler(predictions_config, model_options)
        elif self.model_type == ModelType.ONNX:
            self.model_handler = OnnxModelHandler(predictions_config, model_options)
        else:
            raise ValueError(f"ERROR: Unsupported model type: {self.model_type}")

        return True, model_options

    def get_predictions(self, frame: np.ndarray) -> dict[str, Any] | None:
        if self.model_handler is None:
            return None
        return self.model_handler.get_predictions(frame)

    async def reload_model(self, predictions_config: PredictionsConfig) -> None:
        is_updated_handler, model_options = self._update_model_handler(
            predictions_config
        )

        if not is_updated_handler and self.model_handler is not None:
            await self.model_handler.reload_model(predictions_config, model_options)
