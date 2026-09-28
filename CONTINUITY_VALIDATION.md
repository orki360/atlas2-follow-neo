# Continuous single-target tracking

## Scope

Desktop-only update within version 11.1, identified in new session logs as
`tracking_revision: continuity-1`, `tracking_motion: botsort_only`.
MSDKRemote and the protocol are unchanged. No phone connection or aircraft
commands were used for development or validation. Restart the desktop GUI to
load the update. The original YOLO segmentation model and dependencies remain.

## Implementation

`BoTSORTTracker` no longer derives from the legacy `BBoxTracker`. It retains the
same motion-estimate interface for tracking, spacing, overlay and command gates,
but reads BoT's mean/covariance. BoT velocities are converted from nominal 30Hz
steps to pixels/second. Read-only display projections do not advance the filter.
Legacy headless tests still use the historical tracker for regression comparison;
the live GUI always selects BoT-SORT.

Real measurement timestamps are updated only by actual matched YOLO boxes. No
prediction or unconfirmed candidate refreshes them. Proximity, braking, stale
video/heading gates, command caps, keyboard protections and command smoothing
remain. START DANCE/Apply reset control history without deleting target identity;
the next measurement is required before control may resume. RESET and a frame
resolution change still reset the tracker.

The outer 750ms tracker reset is removed. BoT's finite track pool is distinct
from application target retention. Verified recovery updates the original BoT
track's Kalman state using a real detection, and removes a duplicate tracklet
when present. Logical target identity is separately logged from internal IDs.

Recovery settings and rules:

- Normal loss grace defaults to 450ms, adjustable from 300ms to 1s in the GUI.
  Reliable short coasting is bounded by the existing 650ms limit.
- Nearby candidate geometry uses the current camera-compensated estimate,
  relative size, uncertainty, and a bounded center-distance gate. A fixed IoU
  threshold is not the sole recovery criterion. Ambiguous candidates are rejected.
- Three strong observations over at least 60ms, or four mixed/weak observations
  over at least 100ms, are required to restore a nearby lost target.
- Nearby weak recovery is limited to a three-second measurement gap. After a
  longer/geometrically displaced loss, a unique strong candidate may restore the
  single known target after at least eight observations spanning 250ms, with
  size and temporal consistency checks. Persistent false-positive NEO detections
  can still fool this policy; appearance ReID is not enabled.
- Candidate motion pause is bounded to 350ms per candidate encounter and cannot
  extend the search's existing deadline. Accepted detections pause search while
  reacquisition is confirmed. No translation follows unconfirmed candidates.
- BOOST is yaw-only, at most 350ms and bounded by the angular envelope. It needs
  recent supported outward motion near the left/right image edge. Ordinary SCAN
  uses the normal yaw cap, reverses at the configured half-span boundaries, and
  requires fresh heading telemetry. Angles are total spans (20 means +/-10).

Long-loss reacquisition is explicitly for the current one-drone workflow. It
does not assert biometric/object identity or make multiple visually similar
drones interchangeable. A stable ID alone is not validation of correct tracking.

## Replay method

`replay_continuity.py` compares original and updated controllers on the same
logged YOLO detections and decoded clean-video frames from the four requested
recordings. Only exact matches between recording frame IDs and logged inference
frame IDs are replayed. Original decision timestamps and heading telemetry are
used. The tracker is initialized at each clip's beginning; GUI reset events and
the historical state before the clip are not reconstructed. Therefore these are
paired offline comparisons, not an exact reconstruction of the live session.

YOLO is not rerun, and the replay does not simulate how different flight commands
would change future camera images. Accepted-result and search counts are
continuity measures, not ground-truth tracking accuracy. No claim of flight
validation is made. Representative reassociations must also be visually checked.


## Results

| Recording | Matched frames | Accepted measurements: before -> after | Search results: before -> after | Internal ID changes: before -> after |
|---|---:|---:|---:|---:|
| recording_125710_b80325 | 1595 | 843 -> 903 | 447 -> 296 | 16 -> 0 |
| recording_125845_a4a8ea | 758 | 623 -> 623 | 112 -> 109 | 0 -> 0 |
| recording_130039_ddfa39 | 1231 | 470 -> 578 | 481 -> 323 | 15 -> 0 |
| recording_130204_2d805b | 2080 | 718 -> 885 | 296 -> 179 | 13 -> 0 |

The full release gate passed 174 tests, including new continuity tests and
existing spacing/coast behavior exercised on the BoT-only backend. A hidden
native Tk GUI smoke check passed, including version 11.1, the loss-grace
field and Apply / keep target. The GUI was not connected to a phone.

Six sampled recovery frames were inspected visually in recovery_review.jpg:
nearby strong, weak, and longer-loss recovery examples showed the detected
NEO and corrected BoT estimate at corresponding image locations. This is
a limited visual check, not exhaustive ground-truth annotation. Both weak
and strong false positives remain possible.

Raw replay summaries: baseline_final.json and release_replay.json. Full
test output: continuity_test_results.txt. Per-frame development replay data
remain in the separate continuity_update workspace. The recording files
and original session logs were read only.
