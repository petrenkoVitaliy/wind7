from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelType(Enum):
    YOLO = 1
    ONNX = 2


@dataclass(frozen=True)
class ModelConfig:
    size: int
    name: str
    description: str
    path: Path
    model_type: ModelType


MODEL_BASE_PATH: Path = Path("models")


class ModelsConfig(Enum):
    S26_PT_640 = ModelConfig(
        size=640,
        name="S26_PT_640",
        path=MODEL_BASE_PATH / "best.pt",
        model_type=ModelType.YOLO,
        description="YOLO|Raw|S|640",
    )
    S26_PT_800 = ModelConfig(
        size=800,
        name="S26_PT_800",
        path=MODEL_BASE_PATH / "best.pt",
        model_type=ModelType.YOLO,
        description="YOLO|Raw|S|800",
    )
    S26_ENGINE_800_3060 = ModelConfig(
        size=800,
        name="S26_ENGINE_800_3060",
        path=MODEL_BASE_PATH / "best_rtx3060.engine",
        model_type=ModelType.YOLO,
        description="YOLO|TensorRT|S|800|Half|RTX3060",
    )
    S26_OPENVINO_640 = ModelConfig(
        size=640,
        name="S26_OPENVINO_640",
        path=MODEL_BASE_PATH / "best_openvino_model_640_s",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|S|640",
    )
    S26_OPENVINO_640_HALF = ModelConfig(
        size=640,
        name="S26_OPENVINO_640_HALF",
        path=MODEL_BASE_PATH / "best_openvino_model_640_s_half",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|S|640|Half",
    )
    S26_OPENVINO_640_INT8 = ModelConfig(
        size=640,
        name="S26_OPENVINO_640_INT8",
        path=MODEL_BASE_PATH / "best_openvino_model_640_s_int8",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|S|640|INT8",
    )
    S26_OPENVINO_800 = ModelConfig(
        size=800,
        name="S26_OPENVINO_800",
        path=MODEL_BASE_PATH / "best_openvino_model_800_s",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|S|800",
    )
    S26_OPENVINO_800_INT8 = ModelConfig(
        size=800,
        name="S26_OPENVINO_800_INT8",
        path=MODEL_BASE_PATH / "best_openvino_model_800_s_int8",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|S|800|INT8",
    )
    N26_OPENVINO_800 = ModelConfig(
        size=800,
        name="N26_OPENVINO_800",
        path=MODEL_BASE_PATH / "best_openvino_model_800_n",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|N|800",
    )
    N26_OPENVINO_640 = ModelConfig(
        size=640,
        name="N26_OPENVINO_640",
        path=MODEL_BASE_PATH / "best_openvino_model_640_n",
        model_type=ModelType.YOLO,
        description="YOLO|OpenVINO|N|640",
    )
    S26_ONNX_800 = ModelConfig(
        size=800,
        name="S26_ONNX_800",
        path=MODEL_BASE_PATH / "best_onnx_800_s.onnx",
        model_type=ModelType.ONNX,
        description="ONNX|S|800",
    )
    S26_ONNX_640 = ModelConfig(
        size=640,
        name="S26_ONNX_640",
        path=MODEL_BASE_PATH / "best_onnx_640_s.onnx",
        model_type=ModelType.ONNX,
        description="ONNX|S|640",
    )
