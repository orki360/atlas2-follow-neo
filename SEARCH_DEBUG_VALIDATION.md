# Version 11.1 / search-debug-5

## Recorded fault and correction

In session `20260929_121324_fa7092`, clip `recording_121727_3fe020`,
the 00:32 observation compared decision time 184624.375 with heading request
time 184624.390. The heading was incorrectly treated as future data and the
search ended with approximately 9.7 seconds remaining. The session now reads
the heading snapshot first, then captures a new decision time after tracker
processing. Strict telemetry freshness validation remains enabled.

At 01:37.047 the control ACK deadline expired. No further motion attempts were
logged during the remainder of the clip, despite perception reporting search.
Recordings now carry the control snapshot and show inactive authority explicitly.
Search cancellation/re-enable does not replay an old arc. This does not diagnose
the underlying network/phone delay or change the Android protocol.

## Search contract

- Last accepted box centre in [25%, 75%] of image width: symmetric half-angle
  boundaries. Otherwise: full angle toward the frozen estimated loss direction.
- The configured angle is measured from the loss heading, including any BOOST
  motion. Directional boundaries are [0, angle] or [-angle, 0].
- Heading feedback controls reversal and braking. The maximum search command is
  normalized yaw 1.0; this is not a calibrated degrees/second value. Normal manual
  and tracking commands continue to use the user's axis sliders.
- Candidate verification and short telemetry gaps pause with zero movement and
  preserve the same deadline. Heading absence for 350 ms cancels the episode.
- Automatic time is 1 s plus planned travel / 24 degrees per second, at least
  the requested duration and at most 10 s. It is a planning estimate. Timeouts
  explicitly report incomplete coverage instead of claiming the arc was flown.

## Operator workflow

Connect the GUI session and keyboard/control to the same phone. Enable (E).
In Control / Search, set angle/time and press Apply search. Choose LEFT, RIGHT
or CENTER, then START SEARCH TEST. Dance is switched off; only yaw is requested.
The test ends after reaching both measured boundaries once, or on timeout.
STOP SEARCH TEST stops the test while retaining manual authority. Q/Esc releases
authority. Movement keys cancel the test and return to normal manual limits.

Each test creates `logs/<session>/search_test_<id>/events.jsonl`. It includes
settings, anchor heading, target/actual relative angles, angular rate, requested
yaw, actual transport commands and ACKs. The summary includes measured travel,
extrema, overshoot, coverage, stop reason and measurement completeness. Heading
wraparound is unwrapped. A two-second neutral tail measures residual rotation;
new movement, connection loss or shutdown may interrupt it and is marked.

## Local verification

The release gate includes deterministic timestamp ordering, left/right/central
arcs, loss-anchor preservation, braking, candidate/heading pauses, reacquisition,
authority loss, debug start guards, measured wraparound and post-stop drift,
dedicated logs and disconnected recording overlays. Socket-pair fixtures check
actual serialized full-scale yaw, zero translation, normal manual caps and
neutralization when STOP arrives while an ACK is pending. No phone or aircraft
is contacted. Actual hidden Tk widgets are checked using the project interpreter.

Physical stopping distance and target reacquisition at the increased search
speed still require a flight test. The braking rule is a conservative heuristic,
not a guarantee of zero overshoot. Logs report measurement gaps explicitly.
