from server.model_configs import ModelConfig, ModelType, ModelsConfig
from server.predictions_config import PredictionsConfig
from server.model_handler.yolo_model_handler import YoloModelHandler
from server.utils import tprint


class ModelController:
    def __init__(self, predictions_config: PredictionsConfig):
        self.model_type = None
        self.model_handler = None

        self._update_model_handler(predictions_config)

    def _update_model_handler(self, predictions_config: PredictionsConfig) -> tuple[bool, ModelConfig]:
        model_options = ModelsConfig[predictions_config.model_name].value

        if model_options.model_type == self.model_type:
            tprint(f"RELOAD: Same model type: {model_options.model_type}")
            return False, model_options

        tprint(f"RELOAD: New model type: {model_options.model_type}")

        self.model_type = model_options.model_type

        if self.model_type == ModelType.YOLO:
            self.model_handler = YoloModelHandler(
                predictions_config, model_options)
        else:
            raise ValueError(
                f"ERROR: Unsupported model type: {self.model_type}"
            )

        return True, model_options

    def get_predictions(self, frame):
        return self.model_handler.get_predictions(frame)

    async def reload_model(self, predictions_config: PredictionsConfig):
        is_updated_handler, model_options = self._update_model_handler(
            predictions_config)

        if not is_updated_handler:
            await self.model_handler.reload_model(predictions_config, model_options)
