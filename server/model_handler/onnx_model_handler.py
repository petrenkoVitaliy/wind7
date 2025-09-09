import time
import cv2
import asyncio
import numpy as np
import onnxruntime as ort
from scipy.spatial import distance
from collections import OrderedDict

from server.model_configs import ModelConfig
from server.predictions_config import PredictionsConfig
from server.utils import tprint


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            input_centroids[i] = (int((startX + endX) / 2.0),
                                  int((startY + endY) / 2.0))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        result_ids = []
        for i in range(len(input_centroids)):
            matched_id = -1
            for obj_id, centroid in self.objects.items():
                if np.array_equal(centroid, input_centroids[i]):
                    matched_id = obj_id
                    break
            result_ids.append(matched_id)

        return result_ids

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]


class OnnxModelHandler:
    def __init__(self, predictions_config: PredictionsConfig, model_options: ModelConfig):
        self.predictions_config = predictions_config
        self.model_options = model_options
        self.session = self._init_session()
        self._load_metadata()
        self.tracker = CentroidTracker()

    def _init_session(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        return ort.InferenceSession(str(self.model_options.path), sess_options=sess_options, providers=providers)

    def _load_metadata(self):
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def get_predictions(self, frame):
        return self._run_inference(frame)

    async def reload_model(self, predictions_config, model_options):
        self.predictions_config = predictions_config
        self.model_options = model_options

        try:
            loop = asyncio.get_event_loop()
            self.session = await loop.run_in_executor(None, self._init_session)
            self._load_metadata()
            tprint(
                f"RELOAD::ONNX: successfully: {predictions_config.model_name}")
        except Exception as e:
            tprint(f"ERROR::ONNX Failed to reload model: {e}")

    def _run_inference(self, frame):
        t_start = time.perf_counter()
        img_h, img_w = frame.shape[:2]
        imgsz = self.model_options.size

        img_input, r, dw, dh = self._letterbox_fast(frame, (imgsz, imgsz))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            img_input.transpose(2, 0, 1), dtype=np.float32)
        blob /= 255.0
        blob = np.expand_dims(blob, axis=0)

        t_pre = time.perf_counter()
        results = self.session.run(self.output_names, {self.input_name: blob})
        t_inf = time.perf_counter()

        predictions = results[0][0]
        protos = results[1][0]

        conf_threshold = self.predictions_config.conf
        valid_mask = predictions[:, 4] >= conf_threshold
        valid_preds = predictions[valid_mask]

        final_data = []

        if len(valid_preds) > 0:
            boxes = valid_preds[:, :4]
            scores = valid_preds[:, 4]
            classes = valid_preds[:, 5].astype(int)
            mask_coeffs = valid_preds[:, 6:]

            num_masks = len(valid_preds)
            proto_c, proto_h, proto_w = protos.shape

            proto_reshaped = protos.reshape(proto_c, -1)
            raw_masks = np.dot(mask_coeffs, proto_reshaped).reshape(
                num_masks, proto_h, proto_w)

            scaled_boxes = []
            for i in range(num_masks):
                bx1, by1, bx2, by2 = self._scale_coords(
                    boxes[i], r, dw, dh, img_w, img_h)
                scaled_boxes.append([bx1, by1, bx2, by2])

            tracked_ids = [-1] * num_masks
            if self.predictions_config.task == "track":
                tracked_ids = self.tracker.update(scaled_boxes)

            for i in range(num_masks):
                bx1, by1, bx2, by2 = scaled_boxes[i]
                bw, bh = bx2 - bx1, by2 - by1
                if bw < 2 or bh < 2:
                    continue

                scale_p = proto_w / imgsz
                mx1, my1 = int(
                    boxes[i][0] * scale_p), int(boxes[i][1] * scale_p)
                mx2, my2 = int(
                    boxes[i][2] * scale_p), int(boxes[i][3] * scale_p)

                m_crop = raw_masks[i, my1:my2, mx1:mx2]
                if m_crop.size == 0:
                    continue

                m_resized = cv2.resize(
                    m_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)

                binary_mask = (m_resized > 0).astype(np.uint8)

                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS if self.predictions_config.retina_masks else cv2.CHAIN_APPROX_SIMPLE)

                norm_mask = []
                if contours:
                    largest = max(contours, key=cv2.contourArea)

                    pts = largest.reshape(-1, 2).astype(np.float32)
                    pts[:, 0] = (pts[:, 0] + bx1) / img_w
                    pts[:, 1] = (pts[:, 1] + by1) / img_h
                    norm_mask = pts.tolist()

                final_data.append({
                    "box": [float(bx1/img_w), float(by1/img_h), float(bx2/img_w), float(by2/img_h)],
                    "mask": norm_mask,
                    "label": str(classes[i]),
                    "id": tracked_ids[i],
                    "conf": round(float(scores[i]), 2)
                })

        t_post = time.perf_counter()

        return {
            "data": final_data,
            "metrics": {
                "pre": round((t_pre - t_start) * 1000, 1),
                "inf": round((t_inf - t_pre) * 1000, 1),
                "post": round((t_post - t_inf) * 1000, 1),
                "normalize": 0,
                "total": round((t_post - t_start) * 1000, 1)
            }
        }

    def _letterbox_fast(self, img, new_shape):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / \
            2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, dw, dh

    def _scale_coords(self, box, r, dw, dh, img_w, img_h):
        x1 = int(max(0, (box[0] - dw) / r))
        y1 = int(max(0, (box[1] - dh) / r))
        x2 = int(min(img_w, (box[2] - dw) / r))
        y2 = int(min(img_h, (box[3] - dh) / r))
        return x1, y1, x2, y2
