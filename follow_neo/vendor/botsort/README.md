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

BoT-SORT is the sole motion estimator in the live application. BBoxTracker is
retained only for the legacy backend and its historical tests. Recovery applies
verified raw measurements to the original BoT state and retains logical identity.

The stabilization-2 change fixes the batch prediction path: it now calls each
track's `predict()`, which uses the per-engine filter configured for the actual
source-frame interval. The upstream global shared filter used a fixed step and
bypassed that configuration.
