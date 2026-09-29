# Version 11.1 / stabilization-2

This revision addresses the yaw oscillation, recovery and keyboard behaviour
reported in session `20260928_145537_be38f9`. It changes the Python application
only. The installed Android application and command protocol are unchanged.

## Behaviour to check after restarting the GUI

- The header displays `v11.1` and `stabilization-2`.
- Ordinary tracking should reduce yaw as the target approaches the centre;
  forward movement should reduce during misalignment or rapid rotation and
  return gradually. Spacing still stops/brakes near the configured target size.
- A briefly lost target can recover from coherent weak or strong detections
  even when the predicted box has drifted. Yellow remains an unverified
  candidate; it is not a confirmed target. Blue predicted boxes are not YOLO
  measurements. The GUI displays candidate evidence separately from RC ACKs.
- A key release should send neutral promptly without disconnecting a healthy
  connection. A missing reply still causes stop-unconfirmed failure handling.
- Scanning should approach and reverse at both angular boundaries. The new
  **Auto scan time for angle (max 10 s)** checkbox defaults on. A requested
  60-degree / 2-second scan gets 8.1 seconds; turning it off keeps exactly
  2 seconds. This is a planning allowance, not a calibrated angular-speed
  guarantee. Very low yaw caps may not cover the span within ten seconds.

## Validation method

`run_update12_tests.py` includes the historical release gate plus stabilization
regressions for centering, forward attenuation, source-time prediction, candidate
memory, camera transforms, scan timing and socket-level neutral acknowledgements.
Socket tests use local socket pairs; no aircraft/phone connection is made.

`verify_control_gui.py` initializes real hidden Tk widgets and checks the version,
revision and scan-time controls. It never connects video or aircraft control.

Paired replay command (run once per source checkout, with different output paths):

```powershell
python replay_continuity.py --source <checkout> --logs D:/projects/vers1_integration_yam/logs/20260928_145537_be38f9 --output <report.json>
```

The replay uses recorded YOLO boxes/confidences and clean video frames for camera
motion. It pairs exact frame IDs, uses recorded timestamps and heading, and starts
a new tracker at each clip boundary. It does not rerun YOLO, simulate aircraft
dynamics, recover pre-clip tracking history or reproduce every original GUI tick.
Consequently differences from the original live log are expected. Comparisons
between baseline and updated replay use the same input and initialization.

### Paired replay results

Baseline: commit `6cfa920657c0325e1ba322f7adaa6e9f270bac44`.
Across eight clips, 5,550 exact frame IDs were paired. Accepted observations
increased from 2,554 to 2,619. Frames with a strong detection but no accepted
target measurement decreased from 51 to 24. Neither replay changed the selected
BoT target ID. These are association counts, not labelled accuracy measurements.

| Clip | Accepted before / after | Strong detection rejected before / after |
|---|---:|---:|
| 145709 | 0 / 0 | 0 / 0 |
| 145719 | 247 / 247 | 0 / 0 |
| 145728 | 502 / 502 | 1 / 1 |
| 145928 | 816 / 817 | 12 / 12 |
| 150050 | 452 / 501 | 20 / 8 |
| 150135 | 164 / 179 | 18 / 3 |
| 150208 | 0 / 0 | 0 / 0 |
| 150307 | 373 / 373 | 0 / 0 |

In clip 150050, the added continuity includes approximately 7.3-8.1 seconds.
In clip 150135, recovery starts around 6.08 seconds and retains the strong
measurements through 6.55 seconds that baseline rejected. Spot checks of clean
frames with the recovered state box showed the NEO at those locations. This is
not exhaustive labelling of every added match. The extra search-state frames
are expected from the longer auto-time budget and are not additional detections.

Raw replay outputs and sampled frames are retained in the local
`stabilization_update` working folder used to prepare this revision.

Offline command traces cannot establish a straighter physical flight path or
prove the new yaw gains are optimal. Aircraft response, network delay and yaw
caps must still be assessed from the next recorded flight. The 100% edge BOOST
remains distinct from ordinary yaw centering and the normally capped scan.

The server's `success` reply acknowledges its command handler, not physical
aircraft stop. With MSDKRemote unchanged, loss of the computer/network still
cannot be covered by an Android-side command expiry timer in this update.
