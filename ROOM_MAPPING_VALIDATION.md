# 3D room mapping validation — 2026-10-03

## עדכון candidate17 — האצה מאומתת, עדיין יש כשל מעקב

המנוע המותקן כולל התאמת תיאורים ובחירת מאפיינים מהירות יותר, ללא שינוי
תוצאות ההתאמה שנבדקו. 732 בדיקות מנוע ו־26 בדיקות בקר/הזנה מותקנים עברו.
במבחני קצב קלט ICL2 ו־TUM הגיעו לסוף ללא LOST בפריימים שעובדו, אך נותרו
דילוגי תמונות; ICL1 נכשל אחרי פריים 191. המיפוי החי אינו מאומת כיציב.
אין שינוי בהרשאות חומרה, בכיול המשוער או במדיניות ההטסה הידנית.

ראיות עדכניות: `D:/projects/RBD-SLAM-Python/experiments/slam_stability_20261003_candidate17`.
הדוח המלא: [תוצאות ייצוב](D:/projects/RBD-SLAM-Python/docs/stability_report_he.md).
העדכון בגרסה 16 להלן נשמר כהיסטוריה, כולל ניסויים שנכשלו.

## SLAM stability update — 2026-10-03

The application now feeds the existing receiver's raw frames independently of
map snapshots/logging/saving. The selected configuration enables adaptive
keyframe admission with a 100-point support floor; asynchronous mapping stays
OFF. This is an educational manual-mapping setup, not a demonstrated stable
live tracker or flight controller.

Frozen engine candidate16: 541 tests passed, four Windows symlink tests skipped.
192 distinct selected application tests passed across the full run and separate
hidden-GUI checks. The sandbox prevents Tcl initialization; GUI checks therefore
ran outside it with explicit network/subprocess guards and fake receivers. One
transient Tcl startup skip passed in a fresh process. UI-cycle cleanup was kept
on the Tk owner thread in the validation harness; this is disclosed separately
from production changes. No real receiver, control app or aircraft was started.

Actual-controller paced recorded RGB replays include real map snapshots, JSONL
logs and saves. Tracking failures, dropped input and stale poses remain in the
reports. A clean session lifecycle is not tracking success. See the current
[stability report](D:/projects/RBD-SLAM-Python/docs/stability_report_he.md) and
D:/projects/RBD-SLAM-Python/experiments/slam_stability_20261003 for immutable
input/source manifests, raw frame logs, failed variants and test XML.

Earlier sections below retain their historical scope and counts. The latest
camera-profile workflow also permits a labelled, unvalidated Mini 4 Pro
specification estimate; known benchmark calibration does not validate it.

## Delivered scope

The existing GUI gains a **3D Room Mapping** button and a separate interactive
point-cloud window. Live mapping consumes the existing receiver and requires a
measured calibration. The autonomous survey is a labelled simulation of known
room geometry. No new component imports a flight-command sender. No physical
aircraft, motor, or network video session was started during verification.

The requested real-aircraft autonomous room tour is **not complete**. Metric
scale, camera/gimbal/body transforms, reliable observed free space and real
camera calibration remain unresolved. A sparse feature cloud cannot establish
that an unobserved region is safe to traverse.

## Focused tests

Tests ran on the integration application's repaired Python environment:
Python 3.12.14, NumPy 2.0.2, SciPy 1.14.1, OpenCV 4.12.0 and PyAV 15.1.0.
The pytest runner was read from the adjacent SLAM project's environment; its
numerical packages did not replace the application's installed packages.

```text
tests/test_mapping_panel.py
tests/test_room_calibration.py
tests/test_room_mapping_controller.py
tests/test_room_mapping_gui.py
tests/test_room_survey_sim.py
58 passed, 8 subtests passed in 10.94s
```

Coverage includes real hidden Tk widget callbacks, independent projection tests,
bounded 3D point display, frame-source reuse with socket connections forbidden,
map persistence, nonblocking Stop/Save, stale/stop pose invalidation, known-camera
calibration recovery, bad/degenerate calibration rejection, and continuously
checked simulated paths with nonzero altitude changes and occlusion.

Windows shared-temp ACLs initially blocked pytest fixtures. `tests/conftest.py`
now creates a unique project-local base. GUI fixtures use a shorter path because
nested temporary run paths exceeded Windows MAX_PATH. Those failures were in
the test setup, and the focused tests passed after correcting the fixtures.

## Existing regression failures

The initial full-suite run exposed 80 failure entries (including subtests),
375 passed, one skipped and 27 temporary-directory fixture errors. All 80
assertion failures were reproduced on the **unchanged original project**:
80 failed, 345 passed, 182 subtests passed, zero fixture errors in 26.41 seconds.
XML comparison matched all failure identities and multiplicities across 28
test identities, with no additions or removals. They concern preexisting
tracking/spacing/dance/search expectations; these algorithms were not changed.
The 27 fixture errors were resolved for the new tests as described above.
The full existing suite is not claimed green.

## Simulation result and limits

Default geometry is a known 6 × 5 × 3 m synthetic room and an opaque box. A run
selects views based on unseen sampled-surface gain and travel/rotation cost.
The tested run took 40 steps, observed 867 of 911 surface samples (95.17%),
stored nine keyframes and changed altitude by 0.9 simulated meters.
These numbers are not accuracy or coverage measurements of a real room.
The planner knows simulator geometry; camera rotations are instantaneous.
It is a test environment for view selection, not a validated Mini 4 Pro controller.

The command `.venv\Scripts\python.exe run_lab.py --check-room-mapping` exercises
the actual GUI callback in hidden windows and saves `check.json`, `preview.png`,
`view.json` and `cloud.ply` under a new `logs/room_mapping` run. It explicitly
forbids network connections. Live SLAM maps additionally save `map.json`.

The command also passed after deployment in `D:\projects\vers1_integration_yam`
on 2026-10-03. The actual button callback and autonomous simulation completed
with 867 points, nine keyframes, 40 steps and 0.9 m of simulated altitude change.
The checked artifacts are under `logs/room_mapping/20261003_010121_529fd2`.
`check.json` records `success: true`, `gui_button_tested: true`,
`live_video_tested: false` and `flight_commands_sent: 0`. The generated
`preview.png` was visually inspected; it shows the synthetic cloud, trajectory,
camera position and a prominent simulation label.

## Environment repair

The old `.venv` pointed at a missing Windows Store Python. The same Python 3.12
environment was rebound to the available 3.12.14 runtime using `venv --upgrade
--without-pip`, preserving installed packages. The original config and scripts
were backed up under `logs/room_mapping_update_backup/runtime-d6df81e9`.
Hidden Tk succeeds outside the restricted sandbox; no additional GUI package
installation was necessary.

## Follow-up integration fixes and rendered-pixel check — 2026-10-03

Three reproducible GUI/controller defects were corrected. Live `view.json` is
now derived from the same saved map revision as `map.json` and `cloud.ply`, even
if tracking advanced after the display snapshot. Its camera position is empty
because the saved archive contains historical keyframes, not a current pose.
A Stop request cannot be overwritten by a delayed running-state snapshot.
Retained-map capacity is now explicitly visible in the panel while tracking
validity remains a separate status. The new regression cases failed before
the fixes and passed afterwards.

The deployed focused suite returned **65 passed, one skipped, 11 subtests
passed in 19.00 s**. The skipped button-callback test hit an intermittent Tcl
initialization error; its isolated rerun then passed in 0.14 s. Thus all 66
focused tests passed across those runs. Reports are
`logs/room_mapping_update_validation/heartbeat-mapping-tests.xml` and
`heartbeat-tk-recheck.xml` in the same directory. This does not resolve or
reclassify the preexisting full-suite failures above.

The additional command is:

```text
.venv\Scripts\python.exe run_lab.py --check-room-mapping-pipeline
```

It uses the actual `Frame`/`LatestValue`, `RoomMappingController`, `MappingSession`
and SLAM implementation with unchanged production settings. Images and camera
calibration come from a synthetic renderer. Socket construction is forbidden;
the receiver holder cannot be started or stopped. The GUI and aircraft are not
opened. Frames are deliberately paced by computation completion.

The deployed run is recorded in `outputs/pipeline-7c605f76/pipeline-check.json`
and `logs/room_mapping_update_validation/heartbeat-pipeline-cli.json`:

- 13 source frames processed once: ten rendered views, one blank view, then
  two different recovery attempts; zero sockets or flight commands.
- Nine of ten initial views tracked; 444 landmarks and four saved keyframes.
  JSON, filtered PLY, and saved-view geometry agree. Seventeen integration
  checks passed, including stopped-worker and invalid-stopped-pose checks.
- The blank image caused tracking loss. **Repeating the last image failed to
  relocalize**; returning to the earlier mapped view (image six) succeeded.
  This limitation is preserved in the report, not treated as full recovery
  robustness. Synthetic-keyframe Sim(3) alignment RMSE was approximately
  0.006003 in the renderer's coordinates; this is not a real metric-scale fix.
- The deployed run had nine fresh initial poses and a longest calculation of
  0.984 s. An earlier staging run took 1.047 s for one calculation and correctly
  produced only eight fresh initial poses under the unchanged one-second
  stale limit. These are paced synthetic measurements, not a real-time claim.

The seven updated/new code and test files were SHA-256 verified after copying.
Prior files are backed up in
`logs/room_mapping_update_backup/heartbeat-141b8c04`. Real camera calibration,
real-video mapping, observed free space, metric scale, camera/body transforms,
and real-aircraft autonomous execution remain unverified or unimplemented.

## Manual mapping guidance and operator gimbal keys — 2026-10-03

The current user scope is manual flight with measured mapping recommendations,
including separately enabled keyboard gimbal pitch. Advice is visible beside
the pilot's video and in the cloud window. The map window can return focus to
the main video without stopping the map. Existing flight keys retain their
axis meanings; new I/K keys request one relative +3/-3 degree pitch step per
physical press. Advice itself has no control bindings or transport imports.

The engine now adds real per-processed-frame diagnostics from its existing ORB
extraction: count, 3x3 image distribution, brightness, saturated fractions,
Laplacian variance, initialization reason and matched displacement in pixels.
No additional ORB pass or tracking-threshold change was introduced. Image
distribution is not room coverage; blur/exposure causes and helpful next views
are heuristics. Stale, stopped, missing-age and missing-measurement conditions
suppress movement hints. A stale or pending frame cannot refresh diagnostics.
The rendered-pixel pipeline test verifies this producer/controller/advisor
contract using an actual blank image, in addition to fixture-based decisions.

MSDKRemote's local `CommandServer.java` accepts one query client at a time.
Gimbal actions therefore share `HeadingReceiver`'s existing TCP 9997 socket;
no competing query/video connection is created. Local DJI SDK 5.14.0 enum
bytecode verified `RELATIVE_ANGLE=0` and `UNKNOWN=65535`, avoiding the invalid
mode in the old example. The payload ignores roll/yaw and uses relative pitch
with a 0.3 s duration. Creating/enabling the facade causes no hardware action.
An explicit focused keypress is required. Flight control enable/takeoff is not
part of the gimbal path. Unsupported actions/limits remain visible SDK errors.

Tests cover one socket owner, fragmented replies, rejected actions, cancellation
before the send boundary, expiry, disable/focus loss, held-key repeats (including
manual Enable clearing flight keys), old acknowledgements, timeout uncertainty
and no retry/replay on reconnection. A committed send may finish after disable;
the GUI does not claim it can recall that action or measure final pitch.
Gimbal keys are disabled during Dance/search, and all hardware tests use fake
transport. Live aircraft/video/gimbal operation was not performed.

Installed application verification: **160 passed, 15 subtests passed in 19.60 s**,
including actual hidden Tk widgets, original keyboard/focus regression tests,
guidance, gimbal transport, controller, calibration, simulation and rendered SLAM.
Report: `logs/room_mapping_update_validation/manual-guidance-tests.xml`.
This selected suite does not replace the full-project baseline above.

Installed SLAM engine: **273 passed, four skipped in 19.38 s**. The four skips
are the existing Windows symlink-privilege cases. Report:
`D:\projects\RBD-SLAM-Python\outputs\validation\manual-guidance-engine-tests.xml`.
The update added 12 engine diagnostic cases, 49 advice cases, 26 gimbal cases,
and seven GUI integration cases; existing tests also cover the touched paths.

Thirteen application files and four engine files were copied with SHA-256
verification. Pre-update versions are in `outputs/backups/manual-guidance-b096ea2e`
under each project. Real camera calibration and real footage are still needed
to validate mapping recommendations on the user's room; real gimbal direction,
range, response and acknowledgement behavior remain hardware-unverified.

## Automatic per-run mapping log — 2026-10-03

Each accepted live-mode or simulation run now opens a dedicated UTF-8
`mapping.jsonl` in its existing output directory. Startup includes mode, units,
actual loaded calibration and SLAM configuration. The worker records sampled
status, processed source/decode timestamps, diagnostics, camera pose validity,
advice and missing information, saves, observed stop requests, exceptions with
tracebacks, and a final status on orderly completion. Samples are approximately
0.5 seconds apart plus observed state/advice changes, not every input or
processed frame; mapping/export work can delay sampling. JSON records have UTC,
Unix and monotonic timestamps and sequential IDs. No image arrays or full map
geometry are placed in the log.

The map window displays the path and any logging error independently of map
status. Every record is flushed. Later disk-write failure disables further log
writes without aborting map processing/export. A failed log open/start leaves
the previous saved run intact. GUI polling uses cached logger status and does
not wait for write/flush. A forced process termination may leave a partial last
line or omit session_end; no power-loss durability or complete-frame trace is
claimed.

Installed application verification: **110 passed, 2 skipped, 11 subtests passed
in 12.00 seconds**. Only two hidden-Tk cases skipped because of a transient Tcl
initialization/read failure; rerunning those exact cases separately yielded
**2 passed in 5.12 seconds**. Thus all **112 selected tests** passed across the
two runs. The new coverage includes 12 writer tests, seven controller/log tests,
and one GUI log-status test. Existing controller, rendered-image SLAM pipeline,
guidance and hidden-GUI tests also passed. Reports:

- `logs/room_mapping_update_validation/mapping-log-tests.xml`
- `logs/room_mapping_update_validation/mapping-log-tk-recheck.xml`

Checks cover strict JSON parsing and Hebrew text, nonfinite values, sequential
records, independent session files, flushed output while running, responsive
status during blocked flush, real blank-image diagnostics, disconnect errors,
map export after a logging write failure, worker-start failure, and preservation
of the previous saved run when a new log cannot open. Hardware connections were
blocked or replaced with fixtures. The SLAM engine code was unchanged; its suite
was not rerun for this application-only change. This remains synthetic/local
validation, not real Mini 4 Pro video or gimbal validation, and does not replace
the full-project baseline limitations recorded above.

Changed application sources/tests were backed up before installation in
`outputs/backups/mapping-log-2352c46a` and copied with SHA-256 verification.

## Mini 4 Pro specification estimate — 2026-10-03

The user explicitly chose an estimated calibration as the next mapping trial.
`Estimate Mini 4 Pro` creates/selects an engine-compatible profile using the
dimensions of a fresh original BGR frame from the already existing receiver.
It neither connects hardware nor starts mapping. The profile assumes standard
lens, 1x zoom, landscape 16:9, no additional crop/stretch/letterboxing, square
pixels and centered principal point. Unsupported aspect ratios are rejected.
Zoom/lens/crop cannot be verified from dimensions alone.

DJI's accessory page reports standard video FOV 75 degrees at 16:9 and photo
FOV 82.1 degrees. Its axis is unspecified; this implementation explicitly
**assumes diagonal** video FOV. It does not reinterpret 82.1 degrees as a
horizontal video FOV or use the 24 mm equivalent directly as pixel focal length.
Five zero distortion coefficients are unmeasured placeholders. Sources and
assumptions are stored with `calibration_kind=estimated` and `validated=false`.
This is an experimental starting model, not measured DJI-stream calibration.

The profile's provenance is displayed beside video and in the map window,
included with the actual numeric calibration in session_start, view.json and
a loadable calibration.json sidecar. The per-run metadata is retained after
the source file changes or another profile is selected. Successful measured
calibrations retain their quality-screening label when exported/reloaded;
foreign or unlabelled files stay unknown. Pose/map units remain arbitrary.
Numeric mapping thresholds and the SLAM engine code were not changed.

Installed verification: **119 passed, 11 subtests passed in 22.16 seconds**, no
skips or failures. Report:
`logs/room_mapping_update_validation/mini4-estimate-tests.xml`.
This includes 40 estimator/metadata tests and 14 integration cases, plus
existing calibration, controller, mapping-log, rendered SLAM, panel and hidden
GUI checks. Tests cover inverse-FOV geometry, image scaling/pixel centers,
invalid dimensions, atomic no-overwrite output, real engine loading, original
frame dimensions, unavailable/stale inputs, busy workers, visible/cached labels,
no automatic start, fixed provenance across logs/exports, and measured/unknown
round trips. All receiver/hardware interactions were fixtures or blocked.

During staging, the longer new calibration sidecar exposed Windows MAX_PATH
failures in five existing tests. Shortening the unique temporary basename
resolved all five on a focused rerun; the installed suite above then passed.
This does not claim support for arbitrary Windows paths beyond OS limits.

Application sources/tests were copied with SHA-256 checks and prior versions
backed up in `outputs/backups/mini4-estimate-68b315f0`. No live camera, gimbal or
flight validation was performed. A measured board calibration is still useful
for accuracy, but the chosen estimate now allows the requested initial mapping
trial without waiting for it. The next missing evidence is actual received
video and mapping results from the room.


## תוצאת בדיקת חדר מיובא — 2026-10-03

בדיקה לא מקוונת של המנוע על ICL-NUIM מצאה שהמיפוי עדיין אינו יציב: 131/881 תנוחות, אובדן מעקב אחרי 5.33 שניות, שגיאת מיקום RMS של כ־30 ס״מ גם לאחר התאמה למסלול אמת. בדיקת מסלולים נפרדת במודל תלת־ממדי ידוע מראש הצליחה ב־9/9 יעדים ללא התנגשות גאומטרית. היא אינה סיור עצמאי הניזון ממפת ה־SLAM. עברו 28 בדיקות של כלי ההערכה והגאומטריה; קוד האפליקציה לא שונה ולא הופעלה חומרה. הקלט סינתטי והכיול של המאגר ידוע; זו אינה בדיקת אומדן מצלמת Mini 4 Pro.

הדוח, הגרפים, הלוגים, המודל וקוד להרצה חוזרת נמצאים ב־`D:\projects\RBD-SLAM-Python\experiments\indoor_validation\README_HE.md`. יש להשתמש בתוצאה זו כבסיס לשיפור המיפוי לפני בדיקת ניווט חי.
