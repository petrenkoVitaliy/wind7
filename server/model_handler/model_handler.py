from abc import ABC, abstractmethod

from server.model_configs import ModelConfig
from server.predictions_config import PredictionsConfig


class ModelHandler(ABC):
    @abstractmethod
    def get_predictions(self, frame):
        pass

    @abstractmethod
    async def reload_model(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        pass
