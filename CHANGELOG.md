# CHANGELOG — Dancing Drones / Follow NEO

## Version 12.0.0 — Straight approach and numerical recovery revision 13 — 2026-10-01

Release summary: improved straight-line approach to NEO, measured lateral-motion
verification, visible approach diagnostics, numerical recovery for BoT-SORT,
and the retained full-circle search option introduced in revision 12.

Operator feedback — 2026-10-01: the user reports that revision
`straight-approach-13` is very stable during their testing.

- Prefer yaw alignment and forward approach. Live ROLL now requires at least
  four consecutive measurement intervals spanning 150 ms, coherent lateral
  displacement after camera compensation, fresh heading, and a nearly aligned,
  slowly turning aircraft. Repeated measurements, frame gaps, camera-only motion
  and unverified predictions cannot establish lateral evidence. Limit lateral
  demand to 0.10 before the existing operator cap; ramp increases and clear it
  immediately when evidence or alignment is lost.
- Keep the configured forward speed and distance/alignment gates. A stationary,
  centered target does not need to start moving before forward approach is allowed.
- Show the forward cap and limiting reason near Axis limits, plus lateral evidence
  status. Record forward/alignment/spacing diagnostics in observations and sent
  commands; preserve the existing actual-command acknowledgement display.
- Use Joseph-form covariance correction in the existing BoT-SORT Kalman filter,
  with symmetry enforcement and a bounded repair for roundoff-scale PSD loss.
  Reject substantially indefinite or non-finite state instead of masking it.
- On a tracking linear-algebra failure, discard the partially updated tracker,
  clear control history and search, output zero, and require fresh target
  confirmation. Log tracking_numerical_reset with the previous ID and frame.
  Other controller exceptions retain ERROR handling and now include a traceback.
- Display revision straight-approach-13. Preserve the 360-degree search option,
  saved speed settings and MSDKRemote.
- Validation: 308 regression tests, a hidden offline Tk check, and replay of
  15,759 recorded detection/warp updates. The original flight exception was not
  reproduced by that replay; numerical stress and injected-failure tests cover
  covariance stability, zero commands and fresh reacquisition after failure.

## Version 12.0.0 — Full-circle search revision 12 — 2026-10-01

- Allow search angles from 0 to 360 degrees in the GUI and persisted settings.
  Side loss scans the full selected angle toward the loss direction; center loss
  divides the angle equally (360 means 180 degrees on each side).
- Keep accumulated heading through compass wraparound in Dance and search tests.
  A directional search still returns across the same arc; it is not endless yaw.
- For angles above 180 degrees, automatic duration may use up to 35 seconds
  (360 degrees plans 31 seconds directional or 23.5 seconds centered). Manual
  search time remains 2–10 seconds; existing arcs up to 180 keep their 10-second cap.
- Limit the initial BOOST envelope to 180 degrees even for larger SCAN ranges;
  retain the 2-second BOOST timeout and all reacquisition, heading and stop gates.
- Validate 360-degree settings, wraparound, both directions, centered coverage,
  return travel, large-arc BOOST handoff and command rejection contracts offline.
- Display revision search-360-12. No changes to approach control, axis speed
  settings or MSDKRemote. Existing saved search angles are not changed.

## Version 12.0.0 — Measured BOOST and stable search braking revision 11 — 2026-10-01

- Replace the fixed 350 ms BOOST exit with an angle/rate-based braking handoff.
  Share the two-second maximum BOOST duration between planner and send gate;
  expiry stops the search rather than continuing full yaw in SCAN. Keep confirmed
  target reacquisition, candidate/heading pauses, control stops and total deadlines.
- Estimate search yaw rate from a 300 ms window of unwrapped heading samples,
  reducing sensitivity to repeated cached angles with new local query timestamps.
  Normal tracking still uses its existing estimator/controller.
- Give SCAN a continuous distance-based speed profile with rate feedback,
  hysteresis during braking and limited upward command changes. Preserve immediate
  zero requests, real reversal pauses and the continuous same-direction handoff.
  Small arcs can start directly in braking SCAN when insufficient BOOST room remains.
- Record the braking parameters, filtered rate, desired rate, braking distance,
  control reason and BOOST exit reason in dedicated test logs; show the rate estimate
  and control reason in Live states. Keep TEST BOOST -> SCAN and STOP unchanged.
- Braking values are initial test settings, not a calibrated aircraft model.
  Validate measured angle, overshoot and reacquisition in flight before further tuning.
- Display revision search-motion-11. MSDKRemote and saved settings are unchanged.

## Version 12.0.0 — Continuous BOOST-to-SCAN handoff revision 10 — 2026-10-01

- Step 1: remove the fixed 150 ms scan-entry pause only when BOOST hands off to
  SCAN in the same direction. Immediately use SCAN's existing yaw calculation.
- Preserve the 350 ms BOOST limit, boundary/rate braking (which can still request
  zero yaw), actual reversal pauses, direct scan entry, search deadlines and stop gates.
- Apply the shared planner change to Dance and TEST BOOST -> SCAN. Keep the same
  live diagram and dedicated logs; metadata identifies revision boost-scan-handoff-10
  for comparison with the boost-scan-test-9 baseline.
- No changes to MSDKRemote, saved settings or target tracking. Angle-based BOOST
  exit is a separate, later step and is not included here.

## Version 12.0.0 — BOOST-to-SCAN diagnostic revision 9 — 2026-10-01

- Add TEST BOOST -> SCAN with LEFT / RIGHT selection beside the existing scan-only
  test. Run one measured search cycle using the applied angle/time, with yaw only,
  Dance off and no target required. Explicit control enable and fresh heading are required.
- Use the production BOOST and SCAN planner and command gates. Preserve the current
  350 ms BOOST limit and 150 ms scan-entry pause for baseline measurements.
- Open Live states on test start; show the test phase, scan pause, requested yaw,
  recent command acknowledgement, measured yaw/rate, elapsed time and BOOST exit reason.
  Keep STOP SEARCH TEST accessible on the diagram tab.
- Write test metadata, phase transitions, telemetry samples, command attempts,
  acknowledgements and the completion summary to the dedicated
  logs/<session>/search_test_<run_id>/events.jsonl. Identify BOOST_SCAN versus SCAN_ONLY.
  Retain the two-second post-stop measurement and existing operator/control stop handling.
- Display revision boost-scan-test-9. No changes to normal Dance timing, tracking,
  saved settings or MSDKRemote.

## Version 12.0.0 — Live state diagram and automatic control recovery — 2026-09-30

- Release the current application as version 12.0.0, using VERSION.py as the
  shared source for the GUI title, visible version label and application metadata.
- Include the Live states diagram with separate Dance, search-planner and
  control-status indications, plus measured search yaw and target yaw.
- Include connection-preserving command pauses and automatic recovery after
  valid pending/neutral acknowledgements and GUI recovery, without another E press.
- Retain the current tracking/search behavior, including the existing BOOST and
  search timing rules. The proposed angle-driven changes are not part of this release.
- Preserve MSDKRemote compatibility and saved user settings.

## Version 11.1 — Shared version source and visible GUI label — 2026-09-28

### Live state diagram revision 8 — 2026-09-30

- Add a default Live states tab beside the video, highlighting the live Dance
  policy and the separate IDLE / BOOST / SCAN / DONE search-planner phase.
- Display control availability, paused/preview states, transition history,
  state reason, spacing phase, measured search yaw, target yaw and recent RC ACK.
  Use current controller decisions independently of the selected video frame.
- Clear stale/disconnected highlights and distinguish search tests from Dance.
  Preserve control authority when focusing the diagram; clear held movement keys.
- Create diagram canvas items once and refresh at no more than 10 Hz. The view
  does not send commands, change tracking/search transitions or write flight logs.
- Display version 11.1 / `live-state-diagram-8`. MSDKRemote is unchanged.

### Automatic control recovery revision 7 — 2026-09-30

- Resume automatically after an acknowledgement delay once all pending replies
  and the separate neutral command are acknowledged, with a fresh GUI heartbeat.
  No additional E press is needed. Keep the existing socket and repeat the
  enable handshake before allowing new commands.
- Require release and a new press for movement keys held or pressed during the
  pause. Retain Dance selection but require fresh decisions/measurements after
  recovery; never replay a queued flight action or restart an interrupted test.
- Preserve explicit Q / Esc release, focus/input-error holds, flight-action
  holds and ambiguous/rejected reply handling. Heartbeat-only pauses recover
  automatically when GUI updates return. Log automatic versus explicit resume.
- Display version 11.1 / `control-auto-resume-7`. No MSDKRemote changes.

### Control pause revision 6 — 2026-09-30

- Keep the control TCP connection open when command acknowledgements exceed
  their deadline. The 250 ms RC threshold now latches a movement pause and sends
  a neutral command; it no longer triggers automatic disable or disconnection.
- Keep late replies paired with their original commands in FIFO order. Resume
  requires a separate neutral acknowledgement, no pending replies, a fresh GUI
  heartbeat and an explicit Enable (E). Never replay old movement or flight actions.
- Pause on unexpected reply bytes and log their raw contents. Ignore empty line
  separators only. An ambiguous reply stream stays connected but requires an
  operator reconnect before movement can resume.
- Pause movement on GUI heartbeat/focus loss while retaining the connection.
  Show CONTROL CONNECTED - MOVEMENT PAUSED in the GUI and prediction recordings.
  Explicit release/disconnect still works; actual peer/network failures are reported.
- Display version 11.1 / `control-pause-6`. No MSDKRemote or protocol changes.

### Search debugging revision 5 — 2026-09-29

- Fix the heading timestamp race found in session `20260929_121324_fa7092`:
  read the heading snapshot before taking the decision time, after tracker work.
  Keep future/stale telemetry checks; briefly pause on missing heading, then
  cancel after 350 ms if telemetry does not recover. Never move while paused.
- Keep candidate verification inside the existing search episode and deadline,
  display `SEARCH_PAUSED`, and immediately rearm loss recovery after confirmed
  reacquisition so an old completion flag cannot override a short detection gap.
- Use the full configured angle toward the estimated loss direction for targets
  last measured outside the central half of the image. For central losses use
  half the angle on either side. Preserve the loss heading anchor across BOOST
  and SCAN; directional scans return to the anchor without crossing to the other
  side. Auto budgets account for the selected route and remain capped at 10 s.
- Allow production search yaw up to 100%, independently of normal axis sliders,
  with heading-rate braking near boundaries. Translation stays zero. Keyboard
  overrides immediately return to the normal axis caps.
- Add START / STOP SEARCH TEST and LEFT / RIGHT / CENTER scenarios in Control /
  Search. The test uses the production planner and command validation, requires
  explicit existing control authority and fresh telemetry, disables Dance and
  completes one measured sweep or stops at the time limit. No target is needed.
- Write dedicated `logs/<session>/search_test_<id>/events.jsonl` files with
  measured heading, relative angle, angular rate, requested/sent commands, ACKs,
  stop reasons and a summary. Measure two seconds of post-stop drift when the
  connection remains available; mark interrupted tails and telemetry gaps.
- Cancel active recovery when control is lost and reset control history on
  enable/release/error without resetting target identity. Show control authority
  and recent acknowledged yaw in prediction recordings, plus search-test angles.
- Display version 11.1 / `search-debug-5`. MSDKRemote, protocol, models and saved
  user settings are unchanged. See `SEARCH_DEBUG_VALIDATION.md` for validation.

### Search direction revision 4 — 2026-09-29

- Choose the first scan side from consistent measured target motion before loss,
  instead of always starting left. If motion is inconclusive, use the last seen
  image side; a centred stationary target retains a deterministic left fallback.
- Freeze the preferred direction for the loss episode. Continue toward the same
  boundary after BOOST when angular room remains. If that boundary has already
  been reached, reverse inside the original span; never recenter or extend the
  configured range merely to continue toward the lost target.
- Skip or end BOOST early when the remaining angle is within a conservative
  heading-rate braking margin (3 degrees plus 0.45 seconds of closing rate).
  Apply the same margin at the BOOST send gate. Keep normal scan yaw caps,
  candidate holds, heading freshness checks and the original shared deadline.
- Show the first search side in the GUI and log its evidence source, braking
  margin and BOOST exit reason. Display version 11.1 / `search-direction-4`.
- Add symmetric left/right regressions and the recorded 01:10.875 loss case.
  See `SEARCH_DIRECTION_VALIDATION.md`. MSDKRemote and user settings are unchanged.

### Single-target acquisition revision 3 — 2026-09-29

- Fix an initial-acquisition deadlock found in session `20260929_105825_992149`:
  the application retained a tentative BoT ID after BoT removed its track,
  while recovery required an already-confirmed target. Strong later detections
  could therefore never reach TRACK and Dance continued sending zero commands.
- Separate tentative acquisition from confirmed identity. Release a failed
  provisional ID without resetting BoT-SORT or changing confirmed-target recovery.
  A new internal track can confirm the same coherent acquisition sequence.
- Require two distinct, geometrically coherent strong measurements for initial
  confirmation. Weak measurements can preserve existing evidence across short
  confidence dips, but cannot initiate or confirm a target. Limit missing-frame
  gaps to 150ms and total initial evidence age to 600ms. Reject ambiguous candidates;
  compensate initial evidence for camera motion without adding a motion filter.
- Keep Dance output zero before confirmation. Show `Verifying target` and
  `Reacquiring target` instead of a generic wait when acquisition is in progress.
- Display revision `single-target-3` beside version 11.1. Preserve the stabilization
  tuning, user configuration, Android application and wire protocol.
- Add regression coverage for the recorded 88.4% -> 77.4% -> 89.5% failure,
  expired/weak/ambiguous evidence, ID replacement, camera motion and Dance gating.
  See `SINGLE_TARGET_VALIDATION.md` for recorded-input replay results.

### Stabilization revision 2 — 2026-09-29

- Replace ordinary live yaw centering with continuous proportional demand,
  measured image-error rate and fresh aircraft-heading rate damping. Brake
  before crossing the image centre; remove the yaw minimum-command floor and
  delayed reversal. Keep a bounded acceleration ramp and pass reversals via zero.
- Reduce positive forward demand when the target is off-centre or the aircraft
  turns quickly. Restore forward demand gradually; preserve spacing/braking gates.
- Recover against the last real target box carried by camera motion, alongside
  the BoT prediction. Permit plausible partial edge boxes and small overlap without
  accepting isolated or ambiguous detections. No second Kalman filter is added.
- Preserve coherent candidate evidence across gaps up to 150ms and confidence
  dips. Broad returns require eight observations over 250ms, including at least
  three strong observations. Limit evidence to a 600ms window and verification
  pause to 800ms per loss; candidate pauses never extend an active scan deadline.
- Fix BoT batch prediction to use each track's configured source-time interval.
- Brake scans approaching angular boundaries and anchor the scan to heading at
  the last detection when available. BOOST considers the box's departing edge,
  requires outward-motion evidence, and retains its separate 350ms maximum.
- Add **Auto scan time for angle (max 10 s)**, enabled by default. The requested
  time is a minimum in auto mode: a 60-degree scan with 2 seconds requested gets
  8.1 seconds. Disable auto to retain the exact configured timeout. Log requested
  and effective time and whether both boundaries were reached; coverage is not
  guaranteed at low yaw caps. Total angle still means half to either side.
- On ordinary keyboard release/reversal during an RC exchange, send neutral
  immediately and require separate FIFO acknowledgements for the old command
  and neutral before resuming. Retain the original 250ms deadline and failure
  handling; healthy delayed replies no longer cause unnecessary disconnection.
- Show **v11.1 / stabilization-2**, candidate evidence and recent acknowledged
  RC values in the GUI. Add centering, candidate rejection, scan coverage and
  neutralization diagnostics. Keep version/revision in `VERSION.py`.
- Leave MSDKRemote, the wire protocol, user configuration, models and installed
  dependencies unchanged. See `STABILIZATION_VALIDATION.md` for offline checks,
  replay results, limitations and the next flight observations.

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
