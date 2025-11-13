from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from server.configs.models import ModelConfig
from server.configs.predictions import PredictionsConfig


class ModelHandler(ABC):
    @abstractmethod
    def get_predictions(self, frame: Any) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def reload_model(
        self, predictions_config: PredictionsConfig, model_options: ModelConfig
    ) -> None:
        pass
