# CHANGELOG — Dancing Drones / Follow NEO

## Version 11.1 — Shared version source and visible GUI label — 2026-09-28

### Continuous single-target tracking — 2026-09-28

- Live tracking now uses BoT-SORT's Kalman state and camera-motion compensation
  for both control and display. Remove the second Kalman filter from the live
  adapter; convert nominal 30Hz velocity units to pixels/second. Keep historical
  legacy-controller tests separate from the live GUI path.
- Preserve the target across gaps instead of resetting the tracker after 750ms.
  Confirm geometrically plausible recovery over distinct frames, including weak
  detections and small/no-overlap boxes. Restore the original BoT track state
  after verification and retire duplicate tracklets, rather than merely hiding
  ID changes behind a display label.
- After a longer loss, a unique strong returning candidate requires at least
  eight observations spanning 250ms. This is a single-target recovery policy,
  not appearance ReID or proof of identity after an arbitrary disappearance.
- Add a configurable loss grace period (default 450ms). Reliable short coasting
  may postpone scan up to 650ms. Candidate verification pauses motion for a
  bounded period. BOOST requires recent outward evidence near a horizontal edge;
  SCAN retains the configured total angular span, heading and timeout gates.
- START DANCE and Apply retain the tracked target while restarting control and
  spacing. Require a new measurement after a control restart. Explicit RESET
  still clears the target. Add separate logical target / BoT ID diagnostics.
- Show measured detections, unmatched candidates and predictions separately.
  Replace misleading `NO YOLO` text with `YOLO UNMATCHED` / `NO DETECTION`.
- Record recovery evidence, candidate scores, camera transform and verified
  reassociation events. Session logs identify this change as `continuity-1`.
- MSDKRemote, its protocol, the YOLO model, and installed dependencies are
  unchanged. See `CONTINUITY_VALIDATION.md` for replay method and limitations.

### Keyboard recovery and target-loss scan fixes

- Apply Windows key releases immediately and reconcile held keys with physical
  keyboard state. Clear stale keys and queued release callbacks on reconnect,
  enable, and release.
- Interrupt delayed RC acknowledgements on release/reversal; enforce a 250ms
  RC acknowledgement deadline. On failure, attempt neutral plus disable, close
  the channel, and explicitly report an unconfirmed stop. Do not reuse late replies.
- Require neutral acknowledgements around enable before marking control active.
- Log keyboard events, command attempts, acknowledgement times, and stop failures.
- Separate a brief 100% yaw catch-up (up to 350ms) from a left/right scan using
  the normal yaw cap. The configured angle is the total span: 20 degrees means
  10 degrees on either side. Both phases share the search timeout and stop on
  stale video/heading or confirmed reacquisition.
- See [CONTROL_SCAN_FIX.md](CONTROL_SCAN_FIX.md) for behavior, validation, and
  the remaining need for an Android-side command-expiry watchdog. This change
  does not establish flight safety or modify the installed Android application.

### Version display

- Added `VERSION.py` at the project root as the single source of the application
  version, currently `VERSION = "11.1"`.
- The package's `__version__`, GUI window title, session logs, and environment
  reports now use that shared value.
- Added a visible **v11.1** label beside the application name in the GUI header.
  The version is visible inside the window, including before connecting video.
- Future version updates require changing only `VERSION.py` for runtime labels
  and logs. Restart the application after updating the version.

## Version 11.0.0 — BoT-SORT integration — 2026-09-28

This release adds BoT-SORT to live drone tracking, using bounding boxes produced
by the existing segmentation model. The change aims to improve detection
association continuity and handling of camera motion. Tracking accuracy in
flight has not yet been compared with the previous version.

### Tracking and detection association

- BoT-SORT runs automatically in the GUI's live video session.
- Added two-stage association for strong and weak detections and a persistent
  ID for the selected target.
- The minimum detection confidence passed to tracking is
  `min(0.10, confidence * 0.5)`. New-target acquisition still uses the higher
  of `confidence` and `new_track_confidence`.
- Weak detections can continue an existing track but cannot start a new track
  or provide its initial confirmation. Initial confirmation requires two
  qualifying measurements.
- While locked, the tracker does not automatically select another track in
  place of the target. After the loss interval set by `reacquire_seconds`,
  tracking resets and permits a new acquisition.
- Appearance-based re-identification (ReID) is disabled.

### Camera motion compensation

- Added background motion estimation using sparse optical flow and RANSAC.
- Strong detection regions are excluded from background feature selection.
  Correspondences undergo forward/backward validation, and the estimated
  geometric transform is checked for plausibility.
- Camera motion is processed at an image width of up to 640 pixels, then the
  transform is mapped back to the original image coordinates.
- When features are insufficient or the estimate is rejected, no camera
  compensation is applied. This condition is logged explicitly, and tracking
  continues using its motion prediction.

### Integration with the existing controller

- BoT-SORT uses its own Kalman filter for association. The existing
  `BBoxTracker` remains a separate smoothing layer for flight control and
  receives only the actual associated measurement.
- BoT-SORT predictions are not treated as new detections and do not refresh
  measurement timestamps.
- Existing measurement freshness, prediction quality, spacing, search, and
  manual-control priority rules are retained. Weak detections do not authorize
  extended predicted forward motion.
- Calculations use source-frame timestamps, including when frames are skipped,
  rather than the GUI refresh rate.
- Historical analysis tools that instantiate `FollowController()` retain the
  previous tracker for comparison. The live session explicitly selects
  `tracker_backend='botsort'`.

### Interface, logging, and dependencies

- Added BoT-SORT and target-ID labels to the tracking display.
- Added `decision.tracking` log fields for the tracking backend, target ID,
  camera compensation status, and number of accepted background matches.
  Session-start events also record the tracking backend.
- Bundled the official BoT-SORT implementation with local adaptations, its MIT
  license, and source attribution in `follow_neo/vendor/botsort`.
- Added and installed `scipy==1.14.1` and `lap==0.5.12` in the project's environment.
- Added integration tests and an offline image-sequence replay tool.

### Validation performed

- **132 tests passed**, including 14 new tracker/controller integration tests.
  The tests also passed after installation in the project directory.
- Coverage includes weak detections, target identity with a competing object,
  loss and reacquisition, duplicate frames, timing, textureless images, and
  synthetic camera motion compensation.
- Ran `yolo26s_seg_best_v1.onnx` on **60 local frames** using the CPU. An
  associated target measurement was accepted in 19 frames. This is a smoke-test
  result, not an accuracy score.
- Median tracking time was approximately 13.28 ms, with p95 at 18.18 ms,
  excluding YOLO inference and initial startup cost. Replay assumed 30 FPS;
  this sequence has no ground-truth annotations or verified capture timestamps.
- No flight testing was performed, and no commands were sent to a drone during
  validation.

### Scope and version numbering

- The segmentation model and its weights are unchanged. BoT-SORT uses bounding
  boxes rather than segmentation masks for association. Masks remain visible
  in the display.
- This release does not add ROI cropping, change model input resolution, or
  retrain the model.
- The update was initially labeled version 12 during development. At the user's
  request, it was renamed **version 11.0.0** in the interface, version identifier,
  and documentation. Test and report filenames containing `update12` are
  retained to preserve links and the original validation records; they describe
  the current version 11.0.0 changes.
- Files from before the integration are backed up in
  `backups/before_update12_20260928_112452`.

Related documents: [Update instructions (Hebrew)](UPDATE12_HE.md),
[Validation report](VALIDATION_UPDATE12.md), [Test output](test_results_update12.txt),
and [Image replay report](replay_results_update12.json).
