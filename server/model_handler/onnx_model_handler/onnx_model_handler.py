from __future__ import annotations

import asyncio
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from server.configs.models import ModelConfig
from server.configs.predictions import PredictionsConfig
from server.model_handler.model_handler import ModelHandler
from server.model_handler.onnx_model_handler.trackers.botsort_tracker import BoTSORT
from server.model_handler.onnx_model_handler.trackers.byte_tracker import ByteTrack
from server.tracker.tracker import Tracker
from server.utils import L, tprint

TRACKERS: dict[str, type[Tracker]] = {
    "bytetrack": ByteTrack,
    "botsort": BoTSORT,
}


class OnnxModelHandler(ModelHandler):
    predictions_config: PredictionsConfig
    model_options: ModelConfig
    session: ort.InferenceSession
    input_name: str
    output_names: list[str]
    tracker: Tracker

    def __init__(
        self, predictions_config: PredictionsConfig, model_options: ModelConfig
    ) -> None:
        self.predictions_config: PredictionsConfig = predictions_config
        self.model_options: ModelConfig = model_options
        self.session: ort.InferenceSession = self._init_session()
        self._load_metadata()
        self._init_tracker()

    def _init_session(
        self, model_options: ModelConfig | None = None
    ) -> ort.InferenceSession:
        opts = model_options or self.model_options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available = ort.get_available_providers()

        providers: list[str | tuple[str, dict[str, Any]]] = []

        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )

        if "OpenVINOExecutionProvider" in available:
            providers.append("OpenVINOExecutionProvider")

        providers.append("CPUExecutionProvider")

        tprint(L.INIT_ONNX_PROVIDERS, providers=providers)

        return ort.InferenceSession(
            str(opts.path), sess_options=sess_options, providers=providers
        )

    def _load_metadata(self) -> None:
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_names: list[str] = [o.name for o in self.session.get_outputs()]

    def _init_tracker(self) -> None:
        tracker_cls = TRACKERS.get(self.predictions_config.tracker, ByteTrack)
        self.tracker: Tracker = tracker_cls()

    def get_predictions(self, frame: np.ndarray) -> dict[str, Any] | None:
        return self._run_inference(frame)

    async def reload_model(
        self,
        predictions_config: PredictionsConfig,
        model_options: ModelConfig,
    ) -> None:
        old_tracker = self.predictions_config.tracker

        if predictions_config.tracker != old_tracker:
            self._init_tracker()
            tprint(L.RELOAD_ONNX_TRACKER, tracker=predictions_config.tracker)

        try:
            loop = asyncio.get_event_loop()
            new_session = await loop.run_in_executor(
                None, self._init_session, model_options
            )
            self.session = new_session
            self.predictions_config = predictions_config
            self.model_options = model_options
            self._load_metadata()
            tprint(L.RELOAD_ONNX_OK, name=predictions_config.model_name)
        except Exception as e:
            tprint(L.ERROR_ONNX_RELOAD, err=e, exc_info=True)

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, int, int, float, float, float]:
        img_h, img_w = frame.shape[:2]
        imgsz = self.model_options.size

        img_input, r, dw, dh = self._letterbox_fast(frame, (imgsz, imgsz))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(img_input.transpose(2, 0, 1), dtype=np.float32)
        blob /= 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, img_h, img_w, r, dw, dh

    def _infer(self, blob: np.ndarray) -> tuple[Any, Any, float, float]:
        t_pre = time.perf_counter()
        outputs: Any = self.session.run(self.output_names, {self.input_name: blob})
        t_inf = time.perf_counter()
        predictions = outputs[0][0]
        protos = outputs[1][0]
        return predictions, protos, t_pre, t_inf

    def _decode_masks(
        self,
        predictions: Any,
        protos: Any,
        img_h: int,
        img_w: int,
        r: float,
        dw: float,
        dh: float,
        frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        imgsz = self.model_options.size
        conf_threshold = self.predictions_config.conf
        valid_mask = predictions[:, 4] >= conf_threshold
        valid_preds = predictions[valid_mask]

        if len(valid_preds) == 0:
            return []

        boxes = valid_preds[:, :4]
        scores = valid_preds[:, 4]
        classes = valid_preds[:, 5].astype(int)
        mask_coeffs = valid_preds[:, 6:]

        num_masks = len(valid_preds)
        proto_c, proto_h, proto_w = protos.shape

        proto_reshaped = protos.reshape(proto_c, -1)
        raw_masks = np.dot(mask_coeffs, proto_reshaped).reshape(
            num_masks, proto_h, proto_w
        )

        scaled_boxes: list[list[float]] = []
        for i in range(num_masks):
            bx1, by1, bx2, by2 = self._scale_coords(boxes[i], r, dw, dh, img_w, img_h)
            scaled_boxes.append([float(bx1), float(by1), float(bx2), float(by2)])

        tracked_ids: list[int] = [-1] * num_masks
        if self.predictions_config.task == "track":
            tracked_ids = self.tracker.update(scaled_boxes, scores.tolist(), frame)

        final_data: list[dict[str, Any]] = []
        for i in range(num_masks):
            entry = self._build_detection(
                i,
                scaled_boxes,
                raw_masks,
                boxes,
                classes,
                scores,
                tracked_ids,
                img_h,
                img_w,
                imgsz,
                proto_w,
            )
            if entry is not None:
                final_data.append(entry)

        return final_data

    def _build_detection(
        self,
        i: int,
        scaled_boxes: list[list[float]],
        raw_masks: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        tracked_ids: list[int],
        img_h: int,
        img_w: int,
        imgsz: int,
        proto_w: int,
    ) -> dict[str, Any] | None:
        n_bx1, n_by1, n_bx2, n_by2 = scaled_boxes[i]
        bw, bh = n_bx2 - n_bx1, n_by2 - n_by1
        if bw < 2 or bh < 2:
            return None

        scale_p = proto_w / imgsz
        mx1, my1 = int(boxes[i][0] * scale_p), int(boxes[i][1] * scale_p)
        mx2, my2 = int(boxes[i][2] * scale_p), int(boxes[i][3] * scale_p)

        m_crop = raw_masks[i, my1:my2, mx1:mx2]
        if m_crop.size == 0:
            return None

        m_resized = cv2.resize(
            m_crop, (int(bw), int(bh)), interpolation=cv2.INTER_LINEAR
        )

        binary_mask = (m_resized > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS
            if self.predictions_config.retina_masks
            else cv2.CHAIN_APPROX_SIMPLE,
        )

        norm_mask: list[list[float]] = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            largest = cv2.approxPolyDP(largest, 0.001 * peri, True)

            pts = largest.reshape(-1, 2).astype(np.float32)
            pts[:, 0] = (pts[:, 0] + n_bx1) / img_w
            pts[:, 1] = (pts[:, 1] + n_by1) / img_h
            norm_mask = pts.tolist()

        return {
            "box": [
                float(n_bx1 / img_w),
                float(n_by1 / img_h),
                float(n_bx2 / img_w),
                float(n_by2 / img_h),
            ],
            "mask": norm_mask,
            "label": str(classes[i]),
            "id": tracked_ids[i],
            "conf": round(float(scores[i]), 2),
        }

    def _run_inference(self, frame: np.ndarray) -> dict[str, Any]:
        t_start = time.perf_counter()

        blob, img_h, img_w, r, dw, dh = self._preprocess(frame)
        predictions, protos, t_pre, t_inf = self._infer(blob)
        final_data = self._decode_masks(
            predictions, protos, img_h, img_w, r, dw, dh, frame
        )

        t_post = time.perf_counter()

        return {
            "data": final_data,
            "metrics": {
                "pre": round((t_pre - t_start) * 1000, 1),
                "inf": round((t_inf - t_pre) * 1000, 1),
                "post": round((t_post - t_inf) * 1000, 1),
                "normalize": 0,
                "total": round((t_post - t_start) * 1000, 1),
            },
        }

    def _letterbox_fast(
        self, img: np.ndarray, new_shape: tuple[int, int]
    ) -> tuple[np.ndarray, float, float, float]:
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        return (
            cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            r,
            dw,
            dh,
        )

    def _scale_coords(
        self,
        box: np.ndarray,
        r: float,
        dw: float,
        dh: float,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        x1 = int(max(0, (box[0] - dw) / r))
        y1 = int(max(0, (box[1] - dh) / r))
        x2 = int(min(img_w, (box[2] - dw) / r))
        y2 = int(min(img_h, (box[3] - dh) / r))
        return x1, y1, x2, y2
