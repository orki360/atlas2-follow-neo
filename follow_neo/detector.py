"""ONNX adapter: raw YOLO detection and multi-class YOLO26s segmentation.

Raw exports use xywh + class scores, optional mask coefficients, and external NMS.
No Ultralytics or torch dependency is needed at runtime.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, replace
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
    """Load a static raw detection or segmentation ONNX export."""

    def __init__(
        self,
        model_path: str | Path,
        settings: DetectorSettings | None = None,
        *,
        cpu_threads: int | None = None,
        compute_mode: str = "CPU",
        device_id: int = -1,
        diagnostics_dir = None,
        target_class_names: tuple[str, ...] | list[str] | None = None,
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
        # The actual ONNX input is authoritative, including rectangular exports.
        shape = list(self.input.shape)
        if (len(shape) != 4 or shape[:2] != [1, 3]
                or any(not isinstance(n, int) or n <= 0 for n in shape)
                or self.input.type != "tensor(float)"):
            raise RuntimeError(f"Expected static float32 NCHW input; got {shape}")
        self.settings = replace(self.settings, input_height=shape[2], input_width=shape[3])
        self._validate_contract()
        self.target_class_ids = None
        if target_class_names is not None:
            wanted = set(target_class_names)
            if not wanted or not wanted.issubset(set(self.class_names.values())):
                raise ValueError(f"Unknown target classes: {target_class_names}")
            self.target_class_ids = {i for i, label in self.class_names.items() if label in wanted}

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
        if set(self.class_names) != set(range(len(self.class_names))):
            raise RuntimeError("Model class IDs must be contiguous from zero")
        if self.metadata.get('end2end', 'False').lower() == 'true':
            raise RuntimeError("End-to-end exports are not supported; use raw class scores")
        self.segmentation = len(self.outputs) == 2
        if len(self.outputs) not in (1, 2):
            raise RuntimeError("Expected detection output and optional mask prototypes")
        mask_channels = 0
        if self.segmentation:
            proto = list(self.outputs[1].shape)
            if (len(proto) != 4 or proto[0] != 1
                    or any(not isinstance(n, int) or n <= 0 for n in proto)):
                raise RuntimeError(f"Invalid segmentation prototype shape: {proto}")
            mask_channels = proto[1]
        self.mask_channels = mask_channels
        channels = 4 + len(self.class_names) + mask_channels
        shape = list(self.output.shape)
        if (len(shape) != 3 or shape[0] != 1
                or any(not isinstance(n, int) or n <= 0 for n in shape)
                or channels not in shape[1:]):
            raise RuntimeError(f"Unsupported ONNX output {shape}; expected {channels} channels")

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
        resized = (frame if (resized_w, resized_h) == (source_w, source_h)
                   else cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR))
        pad_x = (target_w - resized_w) // 2
        pad_y = (target_h - resized_h) // 2
        # Write normalized RGB planes directly: no padded uint8 canvas or RGB
        # intermediate copy. ROI frames of the target size need no resize.
        tensor = np.full((1, 3, target_h, target_w), 114.0 / 255.0, dtype=np.float32)
        for channel in range(3):
            np.divide(resized[:, :, 2-channel], 255.0,
                      out=tensor[0, channel, pad_y:pad_y+resized_h, pad_x:pad_x+resized_w],
                      dtype=np.float32)
        return tensor, {
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
            "source_w": float(source_w),
            "source_h": float(source_h),
        }

    def _rows(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise RuntimeError(f"Unexpected Neo inference result: {values.shape}")
        base = 4 + len(self.class_names)
        # The fallback supports standalone postprocess tests without a session.
        allowed = (base + self.mask_channels,) if hasattr(self, 'mask_channels') else (base, base + 32)
        rows = values[0]
        if rows.shape[0] in allowed:
            rows = rows.T
        if rows.shape[1] not in allowed:
            raise RuntimeError(f"Unexpected decoded Neo result: {rows.shape}")
        return rows

    def postprocess(
        self, raw_output: np.ndarray, transform: dict[str, float], prototypes=None
    ) -> list[dict[str, Any]]:
        rows = self._rows(raw_output)
        class_end = 4 + len(self.class_names)
        class_ids = np.argmax(rows[:, 4:class_end], axis=1)
        scores = rows[np.arange(len(rows)), 4 + class_ids]
        valid = np.isfinite(rows).all(axis=1) & (scores >= self.settings.confidence)
        target_ids = getattr(self, 'target_class_ids', None)
        rows, scores, class_ids = rows[valid], scores[valid], class_ids[valid]
        if rows.size == 0:
            return []
        valid_size = (rows[:, 2] > 0.0) & (rows[:, 3] > 0.0)
        rows, scores, class_ids = rows[valid_size], scores[valid_size], class_ids[valid_size]
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
        kept = []
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            kept.extend(indices[non_max_suppression(
                boxes[indices], scores[indices], self.settings.iou_threshold,
                self.settings.max_detections,
            )].tolist())
        # Display-only detections must not consume the target detection budget.
        ordered = sorted(kept, key=lambda i: float(scores[i]), reverse=True)
        targets = [i for i in ordered if target_ids is None or int(class_ids[i]) in target_ids]
        display_only = [i for i in ordered if target_ids is not None and int(class_ids[i]) not in target_ids]
        kept = targets[:self.settings.max_detections] + display_only[:self.settings.max_detections]
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
                    "class_id": int(class_ids[index]),
                    "tracking_eligible": target_ids is None or int(class_ids[index]) in target_ids,
                    "class_name": "drone_candidate",
                    "model_class_name": self.class_names[int(class_ids[index])],
                }
            )
            # Masks are display-only. Keep the detector BBOX as the tracker /
            # distance contract; never replace its width with polygon area.
            if prototypes is not None and rows.shape[1]>class_end and len(detections)<=3:
                detections[-1]['segments']=self._segments(rows[index,class_end:],prototypes,boxes[index],transform)
        return detections

    def _segments(self,coefficients,prototypes,box,transform):
        import cv2
        proto=np.asarray(prototypes,dtype=np.float32)
        if proto.ndim == 4 and proto.shape[0] == 1:proto=proto[0]
        if (proto.ndim != 3 or proto.shape[0] != len(coefficients)
                or min(proto.shape) <= 0 or not np.isfinite(proto).all()):
            raise RuntimeError('Invalid segmentation prototypes')
        # Threshold logits at zero (sigmoid > .5), only within the kept box.
        logits=np.einsum('c,chw->hw',coefficients,proto,optimize=False)
        mask_h, mask_w = proto.shape[1:]
        pixel_scale = np.array([self.settings.input_width / mask_w,
                                self.settings.input_height / mask_h], np.float32)
        mask=np.zeros((mask_h,mask_w),np.uint8)
        x1,y1,x2,y2=box/np.tile(pixel_scale,2)
        xs=np.arange(mask_w)[None,:];ys=np.arange(mask_h)[:,None]
        mask[(logits>0)&(xs>=x1)&(xs<x2)&(ys>=y1)&(ys<y2)]=255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:4]:
            if cv2.contourArea(contour)<1:continue
            points=cv2.approxPolyDP(contour,.5,True).reshape(-1,2).astype(np.float32)*pixel_scale
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
