# Version 11.1 / search-direction-4

The estimated loss direction now chooses the first scan boundary. Consistent
measured image motion takes priority, followed by the last observed image side.
A stationary centred target has no directional evidence and defaults left.
The estimate is frozen for the search episode. The scan then alternates between
the configured boundaries until reacquisition, a guard, or the existing deadline.

BOOST and SCAN still share the original heading anchor and total angular span.
If BOOST already reached the preferred boundary, SCAN must return inside that
span. It does not create a fresh outward arc beyond the user's configured limit.

## Recorded case near 01:12

Session: `20260929_114445_3a2a43`, clip `recording_114720_50dc37`.
Original total span: 90 degrees, with auto time enabled (10-second effective
budget). At 70.875 seconds the original code started full left BOOST while
already 26.2 degrees left of the loss anchor and turning left at an estimated
42.0 degrees/second. Original heading subsequently reached 55 degrees left,
ten degrees beyond the configured left boundary.

Replaying the recovery component with the recorded measurements, heading and
candidate/stale guards makes the updated code skip BOOST at 70.875 seconds:
only 18.8 degrees remain, inside its 21.9-degree braking margin. The first
nonzero scan demand is left at 71.031 seconds, toward the -45-degree boundary.
The original window contained 12 BOOST observations; the updated replay has zero.
The replay uses the original aircraft trajectory as input. It does not predict
what heading the aircraft would have reached under the new commands.

## Validation

The release gate adds tests for left/right scan priority without BOOST, direction
continuity after BOOST, reached boundaries, stationary-side fallback, motion
priority over image side, early braking, the send guard, candidate holds, timeout,
stale heading, and the recorded left-loss condition. Actual hidden Tk widgets
are checked without connecting to an aircraft or phone.

The braking margin is a conservative planning heuristic, not a measured stopping
model or a guarantee that physical overshoot is eliminated. The next flight log
must verify aircraft response. Acquisition, normal tracking tuning, the Android
application, wire protocol and saved user configuration are unchanged.
