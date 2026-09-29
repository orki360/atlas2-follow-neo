# Version 11.1 / single-target-3

The fix addresses the first acquisition, before target identity has ever been
confirmed. It does not periodically reset a confirmed target. Existing verified
single-target recovery, coasting, yaw damping, spacing and scan limits remain.

In session `20260929_105825_992149`, the first strong observation at
10:59:30.796 had confidence 0.884. The following 0.774 observation caused BoT to
remove tentative track 1. The application retained that ID, with `confirmed=false`,
so neither new acquisition nor confirmed-target recovery could proceed. The
live session recorded 2,462 Dance commands, all zero.

The new acquisition path carries geometric evidence independently of a provisional
BoT ID. Two coherent strong source frames are required; a weak frame may bridge
the sequence but contributes no strong hit. A missing gap above 150ms or evidence
older than 600ms expires the candidate. Multiple plausible candidates hold rather
than grant control. The selected BoT filter remains the only motion estimator.

## Automated checks

The release gate is `python run_update12_tests.py`. Added cases cover the exact
confidence dip, removed IDs, weak-only sequences, candidate expiry, competing
targets, distant replacements, duplicate frames, camera compensation, retention
of confirmed identity, and zero Dance output before successful acquisition.

`verify_control_gui.py` initializes actual hidden Tk widgets, checks version 11.1
and the `single-target-3` label, and opens no aircraft/phone connections.

## Recorded-input replay

Baseline and updated code are replayed against identical logged YOLO detections
and exact clean-video frames from `recording_105905_aa42af`. Camera motion is
recomputed from those frames. The tracker starts at the clip boundary. This is
an offline check of acquisition/control intent, not physical aircraft movement.
The replay does not reproduce all GUI reconnects or control restarts.

The paired replay covered 5,856 exact frame IDs out of 5,900 recorded frames:

| Metric | stabilization-2 baseline | single-target-3 |
|---|---:|---:|
| Accepted observations | 1 | 1,081 |
| TRACK observations | 0 | 1,062 |
| Strong detection present but not accepted | 1,050 | 95 |

The single recorded ID change in the updated replay replaces the failed initial
tentative ID; it is not a change of confirmed target identity. This replay verifies
escape from the acquisition deadlock, not the accuracy of every association or
the aircraft response. Raw results remain in the local `single_target_update`
preparation folder. All 210 automated tests and the native Tk smoke check passed.

After restarting the GUI, check for `v11.1 / single-target-3`. A confidence dip
during first acquisition should show verification/reacquisition and recover when
consistent strong detections return, rather than remain locked on a removed ID.

The separate acknowledgement timeouts in the supplied session are unchanged by
this focused acquisition fix. MSDKRemote and control transport are unchanged.
