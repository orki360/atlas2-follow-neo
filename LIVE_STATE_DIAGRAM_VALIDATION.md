# Live state diagram — revision 8

Version 12.0.0 / live-state-diagram-8 adds a default Live states tab beside the
video. It reads the current controller decision and control snapshot, independent
of the displayed frame, and highlights the policy state and search planner phase.
The graph shows main transitions, not every possible conditional transition.

Display semantics:

- Green: current state with Dance control enabled. Blue: preview/manual/released.
- Amber: movement paused, search verification/hold, or waiting for current data.
- Red: processing error. Stale/disconnected decisions cannot retain a live search
  highlight. Test mode uses the test planner and is explicitly labelled Dance off.
- The panel shows the last observed transition, reason, spacing phase, measured
  search yaw, target and recent RC acknowledgement. An ACK is not measured motion.
- Focusing the diagram keeps control authority, clearing held movement keys just
  like the existing flight toolbar. Leaving the application retains focus pause.

The view creates canvas items once and updates at no more than 10 Hz on the GUI
thread. It sends no commands and changes no tracking/search/control thresholds.
MSDKRemote, models and saved configuration are unchanged.

Validation uses the release suite, an actual hidden Tk smoke test covering state
highlights and stable item counts, and offline GUI previews at normal/minimum
window sizes. Previews use simulated SCAN data, without phone/aircraft connections.
