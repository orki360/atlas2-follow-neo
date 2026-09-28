# Version 11.0.0 validation

The filename retains the provisional numbering; this report belongs to version 11.0.0. See CHANGELOG.md.

- Release gate: 132 tests passed (132 including 14 new BoT-SORT/controller tests).
  Uses real upstream Kalman, LAPJV association and camera-motion estimation.
- Covered: two-measurement confirmation, high/low threshold boundaries, weak
  continuation, locked target vs distractor, missing/duplicate/stale results,
  reset/reacquisition, source-time intervals, blank frames, invalid coordinates,
  independent tracker IDs, and a 25-pixel camera pan of a 20-pixel-wide target.
- Existing control, segmentation, model checksum, frame scheduling, keyboard,
  recording and telemetry contracts remain in the release gate.
- Offline smoke: 60 local JPEG frames with yolo26s_seg_best_v1.onnx on CPU.
  19 frames provided an accepted target measurement. This is NOT accuracy:
  there are no target annotations or verified original capture timestamps.
  Replay assumes 30Hz. Tracker median 13.28ms, p95 18.18ms, excluding first-call
  import cost and YOLO inference. Performance depends on machine and footage.
- No aircraft connection, takeoff, flight command or live flight validation.

See test_results_update12.txt and replay_results_update12.json for raw results.
