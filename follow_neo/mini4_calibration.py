"""Explicitly unmeasured Mini 4 Pro intrinsics for a first mapping experiment.

DJI lists a 75 degree field of view for standard-lens 16:9 video. The source
does not specify its axis, so treating it as a diagonal FOV is an assumption.
This module reads/writes local JSON only and never opens a camera or socket.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from numbers import Integral
from pathlib import Path

from .room_calibration import _write_json


VIDEO_FOV_DEG = 75.0
MAX_DIMENSION = 16384
MAX_METADATA_BYTES = 2 * 1024 * 1024
ESTIMATE_METHOD = "manufacturer_fov_pinhole_estimate"
MEASURED_METHOD = "OpenCV calibrateCamera (5 distortion coefficients)"
SOURCE_URLS = [
    "https://store.dji.com/product/dji-mini-4-pro-wide-angle-lens",
    "https://www.dji.com/mini-4-pro/specs",
]
ESTIMATE_WARNINGS = [
    "Estimated intrinsics: no camera images were used to measure this calibration.",
    "The published 75 degree video FOV is assumed diagonal; DJI does not specify its axis on the cited page.",
    "Distortion is unmeasured. Five zero coefficients disable correction; they do not establish a distortion-free lens.",
    "Use only standard-lens 16:9 video at 1x zoom, without additional crop, letterboxing or wide-angle accessory.",
    "Camera pose and map geometry may be inaccurate. Metric map scale is not established.",
]


def _image_size(width, height):
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           or not 0 < value <= MAX_DIMENSION for value in (width, height)):
        raise ValueError(f"width and height must be integer pixel sizes in 1..{MAX_DIMENSION}")
    width, height = int(width), int(height)
    expected_height = width * 9 / 16
    # Allow common rounded stream sizes such as 854x480, but not an unrelated
    # aspect ratio in a tiny image merely because it is within a single pixel.
    if (width <= height or abs(height - expected_height) > 1.0
            or abs(height - expected_height) / expected_height > 0.01):
        raise ValueError("The Mini 4 Pro estimate requires landscape 16:9 video (at most one pixel of aspect rounding)")
    return width, height


def estimate_mini4_calibration(width, height, *, output_path=None):
    """Return engine-compatible *estimated*, not measured, camera calibration.

    ``width`` and ``height`` are the original received frame dimensions. Pixel
    centers have integer coordinates and image boundaries lie at -0.5 and
    size-0.5. An optional output is published atomically and never overwritten.
    The 24 mm equivalent focal length is provenance only: mixing the still
    photo FOV with a cropped 16:9 video frame would silently invent intrinsics.
    """
    width, height = _image_size(width, height)
    focal = math.hypot(width, height) / (2 * math.tan(math.radians(VIDEO_FOV_DEG) / 2))
    report = {
        "width": width,
        "height": height,
        "K": [[focal, 0.0, (width - 1) / 2],
              [0.0, focal, (height - 1) / 2],
              [0.0, 0.0, 1.0]],
        "distortion": [0.0] * 5,
        "calibration_kind": "estimated",
        "validated": False,
        "warnings": list(ESTIMATE_WARNINGS),
        "provenance": {
            "method": ESTIMATE_METHOD,
            "camera_model": "DJI Mini 4 Pro",
            "source_urls": list(SOURCE_URLS),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_fov_deg": VIDEO_FOV_DEG,
            "fov_axis": "diagonal (assumed)",
            "photo_fov_deg_not_used": 82.1,
            "equivalent_focal_length_mm_not_used": 24.0,
            "image_source": "original unresized 16:9 stream frames",
            "pixel_coordinate_convention": "integer pixel centers; image boundaries at -0.5 and size-0.5",
            "distortion_measured": False,
            "metric_map_scale_established": False,
            "units": {"K": "pixels", "distortion": "dimensionless", "map_scale": "arbitrary"},
            "assumptions": [
                "The manufacturer's 75 degree standard-lens video FOV is diagonal.",
                "Square pixels, zero skew and optical center at ((width-1)/2, (height-1)/2).",
                "Standard lens without wide-angle accessory; 1x zoom.",
                "Landscape 16:9 video without additional crop, stretching or letterboxing.",
                "Zero distortion is a placeholder, not a lens measurement.",
            ],
        },
    }
    if output_path is not None:
        _write_json(output_path, report, overwrite=False)
    return report


def _bounded_value(value, depth=0):
    """Keep relevant provenance small enough for status snapshots and logs."""
    if isinstance(value, str):
        return value[:1000]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return value if math.isfinite(value) else None
        except OverflowError:
            return None
    if depth >= 2:
        return None
    if isinstance(value, list):
        return [_bounded_value(item, depth + 1) for item in value[:16]]
    if isinstance(value, dict):
        return {str(key)[:80]: _bounded_value(item, depth + 1)
                for key, item in list(value.items())[:16]}
    return None


def calibration_info(path):
    """Describe calibration provenance without assuming foreign files measured.

    ``validated`` means the existing measured-calibration quality screening
    passed; it never means real-flight mapping or metric-scale validation.
    Intrinsic matrix validation remains the engine loader's responsibility.
    File and JSON errors propagate so callers can display a useful error.
    """
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError("Calibration JSON is too large for metadata inspection")
    data = json.loads(content.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Calibration JSON must be an object")
    original = data.get("provenance")
    original = original if isinstance(original, dict) else {}
    relevant = (
        "method", "camera_model", "source_urls", "created_at_utc", "video_fov_deg",
        "fov_axis", "photo_fov_deg_not_used", "equivalent_focal_length_mm_not_used",
        "image_source", "pixel_coordinate_convention", "distortion_measured",
        "metric_map_scale_established", "units", "assumptions", "opencv_version",
        "inner_corners", "square_size_m",
    )
    provenance = {key: _bounded_value(original[key]) for key in relevant if key in original}
    raw_warnings = data.get("warnings", [])
    warnings = [value[:1000] for value in raw_warnings[:12] if isinstance(value, str)] \
        if isinstance(raw_warnings, list) else []
    declared = data.get("calibration_kind")
    if declared == "estimated" or original.get("method") == ESTIMATE_METHOD:
        kind, label, validated = "estimated", "Estimated - Mini 4 Pro specification", False
        warnings = list(dict.fromkeys([*ESTIMATE_WARNINGS, *warnings]))[:16]
    elif (declared in (None, "measured") and original.get("method") == MEASURED_METHOD
          and original.get("image_source") == "original unresized stream frames"
          and data.get("success") is True and data.get("validated") is not False):
        kind, label, validated = "measured", "Measured - chessboard quality checks passed", True
        warnings.insert(0, "Calibration quality checks do not establish real-flight map accuracy or metric SLAM scale.")
    else:
        kind, label, validated = "unknown", "Calibration origin or quality unverified", False
        warnings.insert(0, "This file is not identified as a successful calibration from the application's measured-calibration workflow.")
    return {"kind": kind, "label": label, "validated": validated,
            "provenance": provenance, "warnings": warnings[:16]}
