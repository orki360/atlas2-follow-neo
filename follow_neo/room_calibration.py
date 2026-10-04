"""Measured chessboard calibration using OpenCV, not a port of its solver.

Images must come from the original video stream, before overlays or resizing.
No camera, network, or aircraft connection is made by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

MIN_VIEWS = 10
MAX_RMS_PX = 1.0
MAX_VIEW_RMSE_PX = 2.0
MIN_NORMAL_SPAN_DEG = 8.0
MIN_NORMAL_SEPARATION_DEG = 15.0
MIN_PERSPECTIVE_SPAN = 0.025
DUPLICATE_RMSE_PX = 2.0


def _pattern(columns, rows, square_size_m):
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 3
           for x in (columns, rows)):
        raise ValueError("columns and rows must be integers >= 3 (inner corners)")
    if not math.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("square_size_m must be finite and positive")
    lattice = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objects = np.zeros((columns * rows, 3), dtype=np.float32)
    objects[:, :2] = lattice * square_size_m
    return lattice, objects


def _report(width, height, columns, rows, square_size_m):
    return {"success": False, "K": None, "distortion": None,
            "width": width, "height": height, "rms_px": None, "views": 0,
            "per_view_rmse_px": [], "rejectedfiles": [], "warnings": [],
            "quality": {"minimum_unique_views": MIN_VIEWS,
                        "maximum_rms_px": MAX_RMS_PX,
                        "maximum_view_rmse_px": MAX_VIEW_RMSE_PX,
                        "minimum_normal_span_deg": MIN_NORMAL_SPAN_DEG,
                        "minimum_normal_separation_deg": MIN_NORMAL_SEPARATION_DEG,
                        "minimum_perspective_span": MIN_PERSPECTIVE_SPAN},
            "provenance": {"method": "OpenCV calibrateCamera (5 distortion coefficients)",
                           "opencv_version": cv2.__version__,
                           "inner_corners": [columns, rows],
                           "square_size_m": square_size_m,
                           "image_source": "original unresized stream frames",
                           "created_at_utc": datetime.now(timezone.utc).isoformat(),
                           "metric_map_scale_established": False}}


def _fail(report, message):
    report["success"] = False
    report["warnings"].append(message)
    return report


def _write_json(path, report, overwrite):
    """Publish a complete JSON only; exclusive creation protects old calibration."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Calibration already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # Linking within this same directory atomically fails if the target
            # appeared during calibration; no partial JSON is ever published.
            os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def calibrate_observations(image_points, width, height, columns=9, rows=6,
                           square_size_m=0.025, *, output_path=None,
                           overwrite=False, view_names=None):
    """Calibrate detected inner corners; return a quality report, never guessed K.

    Each view is an (rows*columns, 2) or (rows*columns, 1, 2) pixel array.
    Quality failures return success=False and do not write output_path.
    Invalid arguments and output I/O errors raise exceptions. Geometry gates are
    useful screening, not proof of accuracy on unseen real camera frames.
    """
    lattice, objects = _pattern(columns, rows, square_size_m)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0
           for x in (width, height)):
        raise ValueError("width and height must be positive integer pixel sizes")
    report = _report(int(width), int(height), columns, rows, square_size_m)
    observations = list(image_points)
    names = list(view_names) if view_names is not None else [str(i) for i in range(len(observations))]
    if len(names) != len(observations):
        raise ValueError("view_names must match image_points")
    if output_path is not None and Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Calibration already exists: {output_path}")
    accepted, accepted_names, perspectives = [], [], []
    for name, observation in zip(names, observations):
        corners = np.asarray(observation, dtype=np.float32)
        if corners.shape not in ((columns * rows, 2), (columns * rows, 1, 2)):
            raise ValueError(f"Invalid corner array shape for {name}")
        corners = corners.reshape(-1, 2)
        reason = None
        if not np.isfinite(corners).all():
            reason = "nonfinite corners"
        elif np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
            reason = "corners outside image"
        elif any(min(np.sqrt(np.mean(np.sum((corners - old) ** 2, axis=1))),
                     np.sqrt(np.mean(np.sum((corners[::-1] - old) ** 2, axis=1))))
                 < DUPLICATE_RMSE_PX for old in accepted):
            reason = "duplicate view (within 2 pixels)"
        else:
            homography, _ = cv2.findHomography(lattice, corners, method=0)
            if homography is None or not np.isfinite(homography).all() or abs(homography[2, 2]) < 1e-12:
                reason = "degenerate board geometry"
            else:
                perspectives.append(homography[2, :2] / homography[2, 2] * [columns - 1, rows - 1])
        if reason:
            report["rejectedfiles"].append({"file": str(name), "reason": reason})
        else:
            accepted.append(corners)
            accepted_names.append(str(name))
    report["views"] = len(accepted)
    report["provenance"]["accepted_views"] = accepted_names
    if len(accepted) < MIN_VIEWS:
        return _fail(report, f"Need at least {MIN_VIEWS} distinct complete board views; found {len(accepted)}.")
    perspective_span = np.ptp(np.asarray(perspectives), axis=0)
    report["quality"]["observed_perspective_span"] = perspective_span.tolist()
    if np.any(perspective_span < MIN_PERSPECTIVE_SPAN):
        return _fail(report, "Insufficient perspective variation: tilt the board in both directions and vary its position.")
    try:
        rms, K, distortion, rotations, translations = cv2.calibrateCamera(
            [objects.copy() for _ in accepted], accepted, (int(width), int(height)), None, None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10))
    except cv2.error as error:
        return _fail(report, f"OpenCV calibration failed: {error}")
    if not (np.isfinite(rms) and np.isfinite(K).all() and np.isfinite(distortion).all()
            and all(np.isfinite(v).all() for v in [*rotations, *translations])):
        return _fail(report, "Calibration produced nonfinite parameters.")
    errors, normals = [], []
    for corners, rotation, translation in zip(accepted, rotations, translations):
        projected, _ = cv2.projectPoints(objects, rotation, translation, K, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1)))))
        matrix = cv2.Rodrigues(rotation)[0]
        if np.any((objects @ matrix.T + translation.reshape(1, 3))[:, 2] <= 0):
            return _fail(report, "Calibration placed a board behind the camera.")
        normals.append(matrix[:, 2])
    if not np.isfinite(errors).all():
        return _fail(report, "Calibration produced nonfinite reprojection errors.")
    normals = np.asarray(normals)
    angular_span = np.ptp(np.rad2deg(np.arctan2(normals[:, :2], normals[:, 2, None])), axis=0)
    separation = float(np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    report.update(K=K.tolist(), distortion=distortion.reshape(-1).tolist(),
                  rms_px=float(rms), per_view_rmse_px=errors)
    report["quality"].update(normal_span_deg=angular_span.tolist(), max_normal_separation_deg=separation)
    if rms > MAX_RMS_PX or max(errors) > MAX_VIEW_RMSE_PX:
        return _fail(report, "Reprojection error exceeds limits (overall 1 px, each view 2 px). Retake sharp, complete views.")
    if np.any(angular_span < MIN_NORMAL_SPAN_DEG) or separation < MIN_NORMAL_SEPARATION_DEG:
        return _fail(report, "Insufficient board normal diversity: include tilts around both board axes.")
    if not (0.1 * width < K[0, 0] < 10 * width and 0.1 * height < K[1, 1] < 10 * height
            and 0 < K[0, 2] < width and 0 < K[1, 2] < height):
        return _fail(report, "Implausible focal length or principal point; collect better distributed board views.")
    report["success"] = True
    report["warnings"].append("Low reprojection error alone does not establish real-world map accuracy or metric SLAM scale.")
    if output_path is not None:
        _write_json(output_path, report, overwrite)
    return report


def calibrate_directory(images_dir, output_path, columns=9, rows=6,
                        square_size_m=0.025, *, overwrite=False):
    """Find a board in PNG/JPEG/BMP/TIFF frames and calibrate that video mode."""
    _pattern(columns, rows, square_size_m)
    directory = Path(images_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Calibration already exists: {output_path}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower()
                   in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    found, names, rejected, size = [], [], [], None
    mismatch = False
    for path in files:
        try:
            gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except (cv2.error, OSError):
            gray = None
        if gray is None:
            rejected.append({"file": path.name, "reason": "unreadable image"})
            continue
        current = (gray.shape[1], gray.shape[0])
        if size is None:
            size = current
        if size != current:
            mismatch = True
            rejected.append({"file": path.name, "reason": "image resolution mismatch"})
            continue
        ok, corners = False, None
        try:
            if hasattr(cv2, "findChessboardCornersSB"):
                ok, corners = cv2.findChessboardCornersSB(gray, (columns, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                ok, corners = cv2.findChessboardCorners(gray, (columns, rows),
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        except cv2.error:
            ok = False
        if ok:
            found.append(corners)
            names.append(path.name)
        else:
            rejected.append({"file": path.name, "reason": "complete chessboard not detected"})
    width, height = size if size is not None else (0, 0)
    if mismatch or not found:
        report = _report(width, height, columns, rows, square_size_m)
        report["views"] = len(found)
        report["rejectedfiles"] = rejected
        return _fail(report, "All source frames must have identical original resolution." if mismatch
                     else "No complete chessboard views found.")
    # Delay publishing until the complete provenance, including rejected files,
    # has been assembled. Quality failure always leaves an old output untouched.
    report = calibrate_observations(found, width, height, columns, rows, square_size_m,
                                    view_names=names)
    report["rejectedfiles"] = rejected + report["rejectedfiles"]
    report["provenance"]["images_directory"] = str(directory.resolve())
    if report["success"]:
        _write_json(output_path, report, overwrite)
    return report


def write_board_svg(path, columns=9, rows=6, square_size_mm=25, *, overwrite=False):
    """Write a dimensioned vector board with one-square white quiet border.

    Print at actual size (100%, no fit-to-page) and verify a square with a ruler.
    Defaults need a 300 x 250 mm sheet, larger than A4: use A3 or adjust size.
    """
    _pattern(columns, rows, square_size_mm / 1000)
    target = Path(path)
    width, height = (columns + 3) * square_size_mm, (rows + 4) * square_size_mm
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" viewBox="0 0 {width:g} {height:g}">',
             f'<rect width="{width:g}" height="{height:g}" fill="white"/>']
    for y in range(rows + 1):
        for x in range(columns + 1):
            if (x + y) % 2 == 0:
                parts.append(f'<rect x="{(x + 1) * square_size_mm:g}" y="{(y + 1) * square_size_mm:g}" width="{square_size_mm:g}" height="{square_size_mm:g}" fill="black"/>')
    parts.append(f'<text x="{square_size_mm:g}" y="{(rows + 3.5) * square_size_mm:g}" font-family="sans-serif" font-size="4">{columns} x {rows} inner corners; {square_size_mm:g} mm squares. Print 100%; verify with ruler.</text>')
    parts.append('</svg>')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write("\n".join(parts) + "\n")
    return target
