# Bundled BoT-SORT

Source: https://github.com/NirAharon/BoT-SORT
Commit: 251985436d6712aaf682aaaf5f71edb4987224bd
License: MIT, copyright 2022 Nir Aharon; full text in LICENSE.

The upstream association stages, XYWH Kalman filter, track lifecycle and LAPJV
assignment are retained. Local changes:

- Relative imports, NumPy 2 compatibility, no plotting or FastReID dependency.
- ReID disabled explicitly. No weights downloaded by the application.
- NumPy IoU replaces cython_bbox using the same inclusive-pixel convention.
- Preserve the current matched raw box; control never consumes a forecast as a
  fresh measurement. Inclusive high/low threshold boundaries.
- Session instances do not reset the global ID counter. Removed-track history is
  bounded; newly removed tracks are excluded from the lost pool immediately.
- Prediction uses each track's own Kalman instance, with source-time intervals
  in nominal 30Hz units and conservative dt-dependent process noise. Predicted
  width/height are clamped positive.
- The application replaces upstream GMC with follow_neo.camera_motion.CameraMotion:
  sparse optical flow, detection exclusion masks, forward/backward validation,
  RANSAC, transform plausibility checks, and explicit identity fallback.

BoT-SORT selects the target ID and associates measurements. The existing
BBoxTracker remains a separate source-time control smoother, with its existing
uncertainty/quality and age guards. Only real matched measurements feed it.
