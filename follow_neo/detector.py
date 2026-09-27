"""ONNX adapter: one-class YOLOv8 detection and YOLO26s segmentation.

Raw exports use xywh + score, optional 32 mask coefficients, and external NMS.
No Ultralytics or torch dependency is needed at runtime.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectorSettings:
    input_width: int = 640
    input_height: int = 640
    confidence: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 20


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    overlap = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area + other_area - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 1e-9)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Return retained indices in descending confidence order."""

    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while order.size and len(kept) < max(1, int(max_detections)):
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[
            _iou_one_to_many(boxes[current], boxes[remaining]) <= float(iou_threshold)
        ]
    return kept


class OnnxDroneDetector:
    """Load either supported one-class detection or segmentation ONNX export."""

    def __init__(
        self,
        model_path: str | Path,
        settings: DetectorSettings | None = None,
        *,
        cpu_threads: int | None = None,
        compute_mode: str = "CPU",
        device_id: int = -1,
        diagnostics_dir = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.settings = settings or DetectorSettings()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Neo ONNX model not found: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Neo detection requires the optional onnxruntime dependency"
            ) from exc

        from .compute import create_session
        self.session,self.compute_info=create_session(self.model_path,compute_mode,device_id,diagnostics_dir)
        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()
        self.output = self.outputs[0]
        self.metadata = self.session.get_modelmeta().custom_metadata_map or {}
        self.class_names = self._class_names(self.metadata.get("names"))
        self._validate_contract()

    @staticmethod
    def _class_names(raw: str | None) -> dict[int, str]:
        if not raw:
            return {0: "drone"}
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return {0: "drone"}
        if not isinstance(parsed, dict):
            return {0: "drone"}
        result: dict[int, str] = {}
        for key, value in parsed.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result or {0: "drone"}

    def _validate_contract(self) -> None:
        expected = [1, 3, self.settings.input_height, self.settings.input_width]
        if list(self.input.shape) != expected:
            raise RuntimeError(
                f"Unexpected Neo ONNX input {self.input.shape}; expected {expected}"
            )
        shape = list(self.output.shape)
        self.segmentation = shape in ([1,37,8400],[1,8400,37])
        if self.segmentation:
            if (len(self.outputs)!=2 or list(self.outputs[1].shape)!=[1,32,160,160]
                    or len(self.class_names)!=1 or self.metadata.get('end2end','False').lower()=='true'):
                raise RuntimeError('Expected one-class raw segmentation with 32 mask coefficients and prototypes.')
        elif shape not in ([1, 5, 8400], [1, 8400, 5]):
            raise RuntimeError(
                f"Unsupported ONNX output {shape}; expected one-class detect (5 channels) or segment (37 channels)"
            )

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        import cv2

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Neo detector expects a BGR HxWx3 numpy frame")
        source_h, source_w = frame.shape[:2]
        if source_h < 2 or source_w < 2:
            raise ValueError("Neo detector frame is too small")
        target_w = self.settings.input_width
        target_h = self.settings.input_height
        scale = min(target_w / source_w, target_h / source_h)
        resized_w = max(1, int(round(source_w * scale)))
        resized_h = max(1, int(round(source_h * scale)))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_w - resized_w) // 2
        pad_y = (target_h - resized_h) // 2
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        return tensor, {
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
            "source_w": float(source_w),
            "source_h": float(source_h),
        }

    @staticmethod
    def _rows(raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise RuntimeError(f"Unexpected Neo inference result: {values.shape}")
        rows = values[0]
        if rows.shape[0] in (5,37):
            rows = rows.T
        if rows.ndim != 2 or rows.shape[1] not in (5,37):
            raise RuntimeError(f"Unexpected decoded Neo result: {rows.shape}")
        return rows

    def postprocess(
        self, raw_output: np.ndarray, transform: dict[str, float], prototypes=None
    ) -> list[dict[str, Any]]:
        rows = self._rows(raw_output)
        scores = rows[:, 4]
        valid = np.isfinite(rows).all(axis=1) & (scores >= self.settings.confidence)
        rows, scores = rows[valid], scores[valid]
        if rows.size == 0:
            return []
        valid_size = (rows[:, 2] > 0.0) & (rows[:, 3] > 0.0)
        rows, scores = rows[valid_size], scores[valid_size]
        if rows.size == 0:
            return []
        boxes = np.column_stack(
            (
                rows[:, 0] - rows[:, 2] * 0.5,
                rows[:, 1] - rows[:, 3] * 0.5,
                rows[:, 0] + rows[:, 2] * 0.5,
                rows[:, 1] + rows[:, 3] * 0.5,
            )
        ).astype(np.float32)
        kept = non_max_suppression(
            boxes, scores, self.settings.iou_threshold, self.settings.max_detections
        )
        scale = transform["scale"]
        pad_x, pad_y = transform["pad_x"], transform["pad_y"]
        source_w, source_h = transform["source_w"], transform["source_h"]
        detections: list[dict[str, Any]] = []
        for index in kept:
            x1, y1, x2, y2 = boxes[index]
            x1 = float(np.clip((x1 - pad_x) / scale, 0.0, source_w - 1.0))
            y1 = float(np.clip((y1 - pad_y) / scale, 0.0, source_h - 1.0))
            x2 = float(np.clip((x2 - pad_x) / scale, 0.0, source_w - 1.0))
            y2 = float(np.clip((y2 - pad_y) / scale, 0.0, source_h - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": float(scores[index]),
                    "class_id": 0,
                    "class_name": "drone_candidate",
                    "model_class_name": self.class_names.get(0, "drone"),
                }
            )
            # Masks are display-only. Keep the detector BBOX as the tracker /
            # distance contract; never replace its width with polygon area.
            if prototypes is not None and rows.shape[1]==37 and len(detections)<=3:
                detections[-1]['segments']=self._segments(rows[index,5:],prototypes,boxes[index],transform)
        return detections

    @staticmethod
    def _segments(coefficients,prototypes,box,transform):
        import cv2
        proto=np.asarray(prototypes,dtype=np.float32)
        if proto.shape==(1,32,160,160):proto=proto[0]
        if proto.shape!=(32,160,160) or not np.isfinite(proto).all():
            raise RuntimeError('Invalid segmentation prototypes')
        # Threshold logits at zero (sigmoid > .5), only within the kept box.
        logits=np.einsum('c,chw->hw',coefficients,proto,optimize=False)
        mask=np.zeros((160,160),np.uint8)
        x1,y1,x2,y2=box/4
        xs=np.arange(160)[None,:];ys=np.arange(160)[:,None]
        mask[(logits>0)&(xs>=x1)&(xs<x2)&(ys>=y1)&(ys<y2)]=255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:4]:
            if cv2.contourArea(contour)<1:continue
            points=cv2.approxPolyDP(contour,.5,True).reshape(-1,2).astype(np.float32)*4
            if len(points)<3:continue
            if len(points)>128:points=points[::int(np.ceil(len(points)/128))]
            points[:,0]=np.clip((points[:,0]-transform['pad_x'])/transform['scale'],0,transform['source_w']-1)
            points[:,1]=np.clip((points[:,1]-transform['pad_y'])/transform['scale'],0,transform['source_h']-1)
            result.append(points.tolist())
        return result

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        tensor, transform = self._prepare(frame)
        prepared = time.perf_counter()
        outputs = self.session.run([o.name for o in self.outputs], {self.input.name: tensor})
        raw=outputs[0]
        inferred = time.perf_counter()
        detections = self.postprocess(raw, transform,outputs[1] if self.segmentation else None)
        finished = time.perf_counter()
        return {
            "detections": detections,
            "preprocess_ms": (prepared - started) * 1000.0,
            "inference_ms": (inferred - prepared) * 1000.0,
            "postprocess_ms": (finished - inferred) * 1000.0,
            "total_ms": (finished-started)*1000.0,
            "providers": self.session.get_providers(),
        }
