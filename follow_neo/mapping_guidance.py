"""Measured, advisory-only feedback for a human pilot building a sparse map.

Thresholds below are conservative UI heuristics, not SLAM acceptance criteria.
This module has no transport, control binding, or dependence on flight hardware.
Image feature counts cannot establish room coverage, gravity, or free space.
"""
from __future__ import annotations

import math
from numbers import Real


REFERENCE = (
    "Directions refer to the camera image, not aircraft axes. "
    "Free space is unverified; the pilot must check it before any movement. "
    "Feature distribution is not room coverage."
)
LOW_FEATURES = 60
WEAK_INLIERS = 25


def _number(value, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= minimum else None


def _grid(value):
    """Accept nine flat counts or three rows; invalid/missing is not zero."""
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        values = list(value)
        if len(values) == 3:
            values = [item for row in values for item in row]
        values = [_number(item) for item in values]
    except (TypeError, ValueError):
        return None
    if len(values) != 9 or any(item is None for item in values):
        return None
    return values


def _weak_sector(grid):
    """Require substantial contrast and a unique sector, avoiding noisy ties.

    Opposite/adjacent thirds overlap at corners, so differences must be large
    enough to justify a suggestion. A low count only describes this image.
    """
    if grid is None or sum(grid) < 120:
        return None
    sectors = {
        "up": sum(grid[:3]), "down": sum(grid[6:]),
        "left": sum(grid[::3]), "right": sum(grid[2::3]),
    }
    ordered = sorted(sectors.items(), key=lambda item: item[1])
    (direction, low), (_, second) = ordered[:2]
    high = ordered[-1][1]
    if high < 40 or low > 0.40 * high or second - low < max(15, 0.15 * high):
        return None
    return direction, int(low), int(high)


def build_mapping_guidance(status, view=None):
    """Return display-ready advice without changing input or issuing actions.

    ``metrics`` contains only available finite measurements. ``direction`` is
    a display hint (neutral/up/down/left/right), never a command or body axis.
    Old map geometry alone is intentionally insufficient for current guidance.
    """
    status = status if isinstance(status, dict) else {}
    view = view if isinstance(view, dict) else {}
    diagnostics = status.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    state = str(status.get("state", "idle")).lower()
    mode = str(status.get("mode", view.get("source", "live"))).lower()
    tracking = str(status.get("tracking_state", "")).lower().rsplit(".", 1)[-1]
    features = _number(diagnostics.get("num_features"))
    matches = _number(status.get("num_matches"))
    inliers = _number(status.get("num_inliers"))
    age = _number(status.get("frame_age_seconds"))
    processing = _number(status.get("last_processing_seconds"))
    stale_after = _number(status.get("stale_after_seconds"))
    stale_after = stale_after if stale_after and stale_after > 0 else 1.0
    grid = _grid(diagnostics.get("feature_grid"))
    occupied = _number(diagnostics.get("feature_occupied_cells"))
    if occupied is None and grid is not None:
        occupied = float(sum(count >= 5 for count in grid))
    if occupied is not None and occupied > 9:
        occupied = None

    metrics = {}
    for key, value in (
        ("num_features", features), ("num_matches", matches),
        ("num_inliers", inliers), ("frame_age_seconds", age),
        ("last_processing_seconds", processing),
        ("feature_occupied_cells", occupied),
        ("landmarks", _number(status.get("landmarks"))),
        ("keyframes", _number(status.get("keyframes"))),
    ):
        if value is not None:
            metrics[key] = int(value) if key not in ("frame_age_seconds", "last_processing_seconds") else value
    for key in ("grayscale_mean", "dark_fraction", "bright_fraction",
                "sharpness_laplacian_variance", "median_matched_displacement_px"):
        value = _number(diagnostics.get(key))
        if value is not None and (not key.endswith("_fraction") or value <= 1):
            metrics[key] = value
    ratio = inliers / matches if matches and inliers is not None and inliers <= matches else None
    if ratio is not None:
        metrics["inlier_ratio"] = ratio
    unavailable = [label for value, label in (
        (features, "Feature count unavailable"),
        (age, "Frame age unavailable"),
        (inliers, "Pose inlier count unavailable"),
    ) if value is None]

    def result(code, severity, title, action, reason, *, missing=(), direction="neutral"):
        return dict(code=code, severity=severity, title=title, action=action,
                    reason=reason, metrics=metrics, missing=list(missing) + unavailable,
                    reference=REFERENCE, direction=direction)

    if state == "error":
        return result("error", "error", "Mapping error",
                      "Resolve the mapping error before continuing the survey.",
                      str(status.get("message") or "No current reliable mapping result."),
                      missing=("Current mapping result",))
    if state in ("stopping", "stopped", "idle", "complete", "no_reachable_view"):
        if mode == "simulation":
            return result("simulation_stopped", "info", "Simulation is not live mapping",
                          "Review the simulated cloud; it does not describe the real room.",
                          "Simulation is finished or inactive. No aircraft motion is requested.")
        return result("inactive", "info", "Mapping is inactive",
                      ("Start live mapping to receive measured guidance." if status.get("calibration_selected") else
                       "Choose calibration and start live mapping to receive measured guidance."),
                      "A stored cloud does not provide current video or a current camera pose.",
                      missing=("Active mapping session",))
    if mode == "simulation":
        return result("simulation", "info", "SIMULATION guidance",
                      "Observe the simulated survey. Do not follow its path in the real room.",
                      "The planner uses synthetic geometry; no live aircraft direction is inferred.")

    processed = status.get("last_processed_source_frame_id") is not None
    measured = features is not None or matches is not None or inliers is not None
    if not processed and not measured:
        return result("awaiting_frames", "info", "Waiting for a processed video frame",
                      "Wait for fresh video and a completed mapping result.",
                      "No current feature or pose evidence is available, even if an old cloud is visible.",
                      missing=("Processed video frame",))
    if age is not None and age > stale_after:
        slow = processing is not None and processing >= stale_after
        return result("processing_lag" if slow else "stale_video", "warning",
                      "Mapping result is delayed" if slow else "Video result is stale",
                      "Pause survey expansion and wait for a fresh processed frame.",
                      (f"Last result is {age:.2f} s old; processing took {processing:.2f} s. "
                       "Slow processing may contribute; the video source may also be delayed.") if slow else
                      f"Last result is {age:.2f} s old (freshness limit {stale_after:.2f} s). The cause is not established.",
                      missing=("Fresh processed frame",))
    if processing is not None and processing >= stale_after:
        return result("processing_lag", "warning", "Mapping processing is slow",
                      "Pause survey expansion until processing catches up.",
                      f"The last frame took {processing:.2f} s to process. This measures computation time, not missing imagery.",
                      missing=("Timely mapping updates",))
    if status.get("diagnostics_fresh") is False:
        return result("diagnostics_stale", "warning", "Current image diagnostics are unavailable",
                      "Wait for fresh diagnostics before expanding the survey.",
                      "The displayed image measurements no longer describe a current processed frame.",
                      missing=("Fresh image diagnostics",))
    if age is None:
        return result("diagnostics_unavailable", "info", "Frame freshness is unknown",
                      "Wait for a measured frame age before choosing a new survey direction.",
                      "Feature counts and a pose do not establish when those measurements were made.")

    if (features is not None and features < LOW_FEATURES) or (occupied is not None and occupied < 4):
        details = []
        if features is not None:
            details.append(f"{int(features)} features (guidance heuristic: {LOW_FEATURES} or more)")
        if occupied is not None:
            details.append(f"features in {int(occupied)}/9 image cells (at least 5 per cell)")
        dark = metrics.get("dark_fraction")
        bright = metrics.get("bright_fraction")
        sharpness = metrics.get("sharpness_laplacian_variance")
        if dark is not None and dark > 0.60:
            details.append(f"{dark:.0%} dark pixels; low exposure may contribute")
        if bright is not None and bright > 0.60:
            details.append(f"{bright:.0%} bright pixels; overexposure may contribute")
        if sharpness is not None and sharpness < 30:
            details.append(f"low image-detail score {sharpness:.1f}; blur or plain surfaces may contribute")
        return result("low_features", "warning", "More visible texture is needed",
                      "Pause translation; gently reframe toward stationary textured objects while retaining overlap. Check lighting and focus.",
                      "; ".join(details) + ". These thresholds are heuristics, not a confirmed cause.",
                      missing=("Enough well-distributed image features",))

    if tracking in ("lost", "relocalizing"):
        return result("tracking_lost", "warning", "Camera tracking is lost",
                      "Stop expanding the survey. Manually return the camera view to a recognizable mapped area; do not blindly reverse flight.",
                      str(status.get("tracking_message") or "No accepted current camera pose is available."),
                      missing=("Recovered camera pose",))

    initializing = tracking == "initializing" or diagnostics.get("initialization_reason") in (
        "need_features", "need_second_view", "need_matches", "need_baseline")
    if not initializing and inliers is not None and (
        inliers < WEAK_INLIERS or (matches is not None and matches >= 20 and ratio is not None and ratio < .35)
    ):
        reason = f"{int(inliers)} pose inliers (guidance heuristic: {WEAK_INLIERS} or more)"
        if ratio is not None:
            reason += f"; {ratio:.0%} of {int(matches)} matches support the pose"
        return result("weak_tracking", "warning", "Tracking support is weak",
                      "Slow down and keep familiar textured objects visible before extending the map.",
                      reason + ". This is guidance, not a change to SLAM's pose acceptance rules.",
                      missing=("Stronger pose support",))

    if initializing:
        init_reason = diagnostics.get("initialization_reason")
        if init_reason == "need_matches":
            return result("initialization_matches", "warning", "Views need more matching features",
                          "Keep more of the previous view in frame and avoid fast turns or moving objects.",
                          "Initialization has not found enough reliable correspondences between views.",
                          missing=("Reliable matches between initial views",))
        if init_reason == "need_features" or features is None:
            return result("initialization_wait", "info", "Initialization needs more evidence",
                          "Keep a textured stationary scene visible and wait for feature diagnostics.",
                          "There is not enough measured evidence to suggest a translation.",
                          missing=("Initial image feature evidence",))
        return result("initialization_baseline", "info", "A second translated view is needed",
                      "Only after checking free space, make a small manual sideways translation while keeping the same objects visible. Rotation alone is insufficient.",
                      "Depth needs parallax from different camera positions. Direction and travel distance are not inferred from this image.",
                      missing=("Triangulatable initial views",))

    capacity = status.get("mapping_capacity_reason")
    if capacity:
        return result("map_capacity", "warning", "Map growth is limited",
                      "Save the current map and review the mapping capacity settings before extending the survey.",
                      str(capacity), missing=("Capacity for new map observations",))

    if not status.get("pose_valid"):
        return result("pose_unavailable", "warning", "Current camera pose is unavailable",
                      "Wait for tracking recovery before expanding the survey.",
                      "Video features alone do not establish a valid camera pose.",
                      missing=("Valid current camera pose",))
    if age is None or features is None or inliers is None:
        return result("diagnostics_unavailable", "info", "Guidance evidence is incomplete",
                      "Wait for current image and tracking diagnostics before choosing a new survey direction.",
                      "A pose or stored map alone is insufficient to assess present image quality.")

    sector = _weak_sector(grid)
    if sector is not None:
        direction, low, high = sector
        return result("inspect_image_sector", "info", "Inspect a sparsely featured image region",
                      f"If useful stationary detail is visible there, gently pan or tilt toward image {direction}, preserving overlap. Otherwise keep textured objects in view.",
                      f"The {direction} image third has {low} features versus {high} in the strongest third. This does not mean that room region is unmapped or clear.",
                      direction=direction)
    return result("steady", "info", "Tracking has usable support",
                  "After checking free space, use a small manual sideways or vertical translation while retaining familiar objects. Pause to let the map update.",
                  "Current feature and pose measurements support continued observation. No particular direction or room completion level can be inferred.")
