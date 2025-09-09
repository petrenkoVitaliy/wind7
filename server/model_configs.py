from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class ModelType(Enum):
    YOLO = 1
    ONNX = 2


@dataclass(frozen=True)
class ModelConfig:
    size: int
    name: str
    path: Path
    model_type: ModelType


MODEL_BASE_PATH = Path("models")


class ModelsConfig(Enum):
    # YOLO models
    S26_OPENVINO_640 = ModelConfig(
        size=640, name="S26_OPENVINO_640", path=MODEL_BASE_PATH / "best_openvino_model_640_s", model_type=ModelType.YOLO)
    S26_OPENVINO_640_HALF = ModelConfig(
        size=640, name="S26_OPENVINO_640_HALF", path=MODEL_BASE_PATH / "best_openvino_model_640_s_half", model_type=ModelType.YOLO)
    S26_OPENVINO_640_INT8 = ModelConfig(
        size=640, name="S26_OPENVINO_640_INT8", path=MODEL_BASE_PATH / "best_openvino_model_640_s_int8", model_type=ModelType.YOLO)

    S26_OPENVINO_800 = ModelConfig(
        size=800, name="S26_OPENVINO_800", path=MODEL_BASE_PATH / "best_openvino_model_800_s", model_type=ModelType.YOLO)
    S26_OPENVINO_800_INT8 = ModelConfig(
        size=800, name="S26_OPENVINO_800_INT8", path=MODEL_BASE_PATH / "best_openvino_model_800_s_int8", model_type=ModelType.YOLO)

    N26_OPENVINO_800 = ModelConfig(
        size=800, name="N26_OPENVINO_800", path=MODEL_BASE_PATH / "best_openvino_model_800_n", model_type=ModelType.YOLO)
    N26_OPENVINO_640 = ModelConfig(
        size=640, name="N26_OPENVINO_640", path=MODEL_BASE_PATH / "best_openvino_model_640_n", model_type=ModelType.YOLO)

    # ONNX models
    S26_ONNX_800 = ModelConfig(
        size=800, name="S26_ONNX_800", path=MODEL_BASE_PATH / "best_onnx_800_s.onnx", model_type=ModelType.ONNX)
    S26_ONNX_640 = ModelConfig(
        size=640, name="S26_ONNX_640", path=MODEL_BASE_PATH / "best_onnx_640_s.onnx", model_type=ModelType.ONNX)
