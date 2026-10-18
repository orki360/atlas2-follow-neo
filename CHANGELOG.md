# CHANGELOG — Dancing Drones / Follow NEO

## Version 13.0.0 / tracking-recovery-14 — Neutral synchronization and usable slowdown — 2026-10-18

- Diagnose session 194916: tracking paused at recording 00:46.234 after an extra
  unnumbered success reply. The detector and video remained alive. The original
  source of that extra reply cannot be established from the Python logs.
- For an already enabled Dance only, allow one bounded neutral synchronization
  attempt per control connection after a complete unsolicited success. Send a
  fresh neutral then a unique non-action command; the inspected Android reader's
  exact unknown-command echo proves the preceding neutral was processed. Generic
  ACKs remain quarantined. No reconnect, disable or enable is sent by recovery.
  Missing/malformed markers, repeated faults, partial sends, rejected commands,
  operator release, focus/heartbeat loss and manual mode still fail closed.
  After confirmation retain tracking context but require new video and target
  evidence before movement. Add protocol events and offline regression tests.
- Observation interval accepts any finite value >=1 degree with no upper cap;
  an interval >= the remaining sweep still observes its actual arc boundary.
  Observation yaw now accepts 0–100%, capped by the main SCAN limit. Zero is a
  stationary observation. A 360-degree scan at 60% uses Scan yaw limit=60%, not
  merely Observation yaw=60%; its endpoint remains stationary.
- Slowdown uses a bounded window sized for braking and new detector results,
  plus a sampling rate ceiling consistent with the selected observation cap.
  This removes the old fixed 6-degree/12-deg/s bottleneck. Missing heading,
  missing fresh frames, arc boundaries, source-time translation budgets and
  background/identity vetoes remain enforced. Log effective window and rate.
- No MSDKRemote, decoder, model, saved user settings or target guards changed.

## Version 13.0.0 / tracking-recovery-13 — Scan controls and bounded command reductions — 2026-10-18

- Motion exposes observation intervals (30–120 degrees), a dedicated SCAN yaw
  cap (10–100%), stop/slowdown mode, and observation yaw cap (2–20%). Existing
  configurations retain 60-degree stops and the previous scan cap. BOOST and
  normal tracking caps are independent. The interval applies to large scans
  (120 degrees or more); final partial sectors and outer boundaries are retained.
- Slow observations preserve a small yaw command within a bounded angular
  window, requiring a measured rate <=12 deg/s and two fresh detector source
  frames. Missing heading stops motion; missing results stop at the window end.
  Outer arc boundaries still stop. No extension of the original search deadline,
  target identity, Nano bridge or search translation permission.
- Revalidation may offer one component-wise reduction of a still-authorized
  same-target command while its ACK is pending. Never increase an axis, create
  a new direction or renew the ACK deadline. FIFO replies remain paired with
  their original commands. Further reductions or hard vetoes neutralize normally.
  Seed smoothing from the reduced wire values instead of restarting every axis.
- Log search observation transitions, measured angle/rate, fallback, duration,
  scan caps and actual reduced commands. Add offline safety, transport and GUI
  checks. MSDKRemote and decoder are unchanged.

## Version 13.0.0 / tracking-recovery-11 — Matching controls and fresh frame sampling — 2026-10-18

- Expose primary, weak and tentative BoT-SORT matching costs in Settings.
  Defaults remain 0.80 / 0.50 / 0.70; larger values allow looser association.
  Persist validated settings and apply them to the existing tracking engine.
- Log bounded `botsort_association` reports once per inference: thresholds,
  best/matched costs, target IDs, rejected costs and assignment competition.
  Existing class, identity, source-age and Nano/background vetoes are unchanged.
- Replace immediate video-frame rejection with a capacity-one latest-frame
  mailbox and a paced publisher. Conversion/callbacks leave the receive/decode
  worker; early frames may survive until the next publication deadline.
  Never duplicate frames or replace source timestamps; discard pending or
  converted frames older than 100ms and invalidate them on reconnect/stop.
- Add sampler replacement/expiry/age counters and concurrent pipeline-stage
  diagnostics. Codec, decoder options, MSDKRemote and model providers unchanged.


## Version 13.0.0 / tracking-recovery-10 — Search direction and adjustable motion — 2026-10-18

- Time-aware raw NEO association tolerates rapid crossings. Two strong unique
  detections update the remaining search route and retain a yaw-only hint for
  650ms of source time through braking. Original arc/deadline remain bounded;
  partial redirected routes are not reported as complete circle coverage.
- Add a persistent Motion tab: forward rectangle width/height, soft edge,
  yaw demand/rate ceilings, recovery rise, optional Nano support requirement,
  visual age limit, search translation shares/budget and observation timing.
  Render four grey corners in both views and prediction recordings, no masks.
- Sustained fresh real NEO proof can use the wider forward envelope while Nano
  is unavailable. Hard background, class, texture and identity vetoes remain.
- Search may combine yaw with forward and lateral approach to a bearing backed
  by three real measurements and a previously permitted forward command. Default
  expiry is 1s from its original YOLO source, max 1.2s; normal coast remains 550ms.
  Caps remain 0.35 / 0.20 of user axis limits; heading change remains <=35 degrees.
  An explicit edge departure may retain this independent search permission only
  with matching exit context and passing appearance/geometry/texture checks.
  Nano stays blocked; resets, renewed conflicts and source expiry revoke motion.
- Shorten minimum look from 350ms to 150ms and settled confirmation from 120ms
  to 80ms; retain two distinct post-settle source frames and actual yaw-rate gate.
- Log motion settings, limiting factors, retained direction, route revisions,
  exit evidence, search-vector eligibility and expiry in decisions/commands.
- Validation includes offline GUI persistence, guarded-motion/expiry regressions
  and recorded crossing replays at 1:01, 1:32 and 2:02. Replays retain recorded
  camera motion and cannot establish closed-loop flight improvement.
- All 558 release tests passed, including 20 new regressions; the hidden Tk
  verification passed. No flight/phone connection was made. Android ZIP and
  saved configuration SHA-256 match the pre-change snapshot.
- No Android, model, dependency, RC-rate or saved user-configuration changes.

## Version 13.0.0 / tracking-recovery-9 — Source alignment and finite search views — 2026-10-15

- Compare delayed YOLO with actual Nano updates around its source frame, then
  validate displacement to the live frame using three real NEO measurements.
  Historical agreement alone cannot approve live background drift. Preserve
  hard class/identity/geometry/texture/exit vetoes and real-source deadlines.
- Large searches (120 degrees or more) brake at 60-degree views, require low
  measured yaw rate and two post-settle detector frames, and stop after finite
  coverage. Explicit yaw-only GUI diagnostics use timed views without detector
  input. Candidate verification and original time/angle contracts remain active.
- Allow verified tracking to combine moderate yaw with forward/lateral motion;
  retain camera-compensated lateral proof and immediate distance/quality stops.
- Preserve a verified movement vector briefly after loss, rotate it into current
  body axes, and decay to zero within 550ms of its original YOLO source. Cap it
  at 0.35 forward / 0.20 lateral before GUI limits. Recheck live vetoes, sensor
  freshness and source expiry at send/ACK; no renewal by Nano-only frames.
- Log alignment, combined-motion eligibility, recovery vectors and search views.
  Recorded gate replay permits 11 additional positive-forward decisions and
  retains all 561 sampled hard translation vetoes. Known background-drift cases
  remain blocked. This is offline validation, not a claim about flight behavior.
- MSDKRemote, model files, dependencies, stored settings and RC rate unchanged.
- Validation: 538 release tests passed, including 26 new regressions and local
  mock transport checks for source expiry and unequal GUI axis limits. No flight
  or drone connection was performed; Android ZIP SHA-256 is unchanged.

## Version 13.0.0 / tracking-recovery-8 — Preserve recovery and independent real-target control — 2026-10-15

- Separate recent-loss candidate guidance from the main recovery budget. A
  partial reacquisition pauses search without consuming it; expiry resumes the
  remaining scan. A completed/old search still only permits a bounded short look.
- Keep a bounded history of unique, strong, unambiguous real NEO measurements.
  Three coherent observations may preserve YOLO authority through soft Nano
  appearance/flow uncertainty; hard identity, background, texture, geometry,
  frame-exit and overlapping MINI conflicts remain vetoes.
- A fresh, independently supported YOLO measurement retains its control intent
  while Nano validates the new anchor, instead of dropping to the loss-bridge cap.
  Nano-only bridge source deadlines, caps, new-stop precedence and ACK gates stay.
- Permit 150–250ms measured lateral evidence only with healthy guarded Nano
  pixels no older than 100ms, matching identity/generation/geometry, valid camera
  compensation and the existing four-sample motion proof. Revalidate at send and
  while awaiting ACK. No lateral prediction from unverified Nano or blind SCAN.
- Convert grayscale after resizing in camera-motion preprocessing. Preserve its
  working resolution, features and bidirectional flow checks; isolated recorded
  comparison found identical transforms/statuses with modest preprocessing savings.
- Log real-source evidence, guidance scope, independent-anchor handoff and lateral
  freshness proof. MSDKRemote, models, dependencies, GUI caps and RC rate unchanged.
- Validation: 512 release tests passed, including 19 new regressions. Recorded
  guard replay gained 92 positive-forward decisions while retaining all 32 hard
  vetoes in the sample. Search replay resumed bounded left SCAN at 44.8s/73.2s
  through the final command gate. No aircraft connection or flight validation.

## Version 13.0.0 / tracking-recovery-7 — Continuity, bounded reacquisition and video diagnostics — 2026-10-15

- Preserve perception and the original search budget during a known Dance ACK
  pause of at most one second. Movement remains neutral until outstanding replies
  and neutral ACK are resolved. A resumed tracking command needs new video,
  a new decision and a real post-resume target measurement; bounded search may
  resume with new video and fresh heading. Explicit enable, operator/focus loss,
  long pauses, rejected/ambiguous channels and stream invalidation retain resets.
- BOOST's two-second deadline now hands off to a ramped SCAN inside the original
  angle and deadline. Cached BOOST commands still expire at two seconds.
- Two unique strong NEO observations can supply a bounded yaw-only look from
  an idle/completed search after a real target was confirmed. This single-use
  look is limited to 0.8 seconds, at most 30 degrees total span, normal GUI yaw
  caps and fresh heading/video. It does not accept a target or authorize translation.
  Competing detections, missing evidence and revoked contracts cancel it. Image
  velocity compensates at most 150 ms / 20% frame width of processing age.
- Soft Nano appearance doubt may fall back to unique strong, same-source real
  YOLO/BoT evidence with matching identity, generation and geometry. Hard MINI,
  background, texture, exit and disagreement vetoes retain precedence, including
  when soft appearance metrics are also present. Nano-alone deadlines/caps stay.
- Add one-second video_health events from a separate diagnostic thread: incoming
  bytes/gaps/timeouts, parse/decode/BGR/callback stage durations and thread CPU,
  keyframes, process CPU, active stage age and correlated control/heading/log
  status. Snapshots also accompany gaps/errors. No stream bytes are transmitted.
- No MSDKRemote, model, local configuration, RC cadence or ACK timeout changes.
  All 492 release tests passed, including 20 new regressions. Recorded gate replay
  allowed 29 soft-doubt decisions (19 forward) without overriding any of 89 hard
  conflicts. Isolated raw-candidate replays used scripted fresh heading, not a
  flight simulation. Further details are documented in UPDATE13_HE.md.
  An additional controller/BoT/send-gate integration regression also passed.

## Version 13.0.0 / link-recovery-6 — Manual ACK and video recovery — 2026-10-15

- Preserve an acknowledged control enable during automatic recovery on the same
  unambiguous connection, after all pending replies and the neutral ACK arrive.
  Avoid repeated SDK enable requests during takeoff. Explicit enable, incomplete
  initial handshakes, rejected commands and ambiguous streams keep their checks.
  Preserve held-key release, fresh UI, flight-action hold and new Dance evidence.
- Retry video connection failures, EOF and the existing 8-second receive/decode
  timeout with 0.5–4-second backoff. Close the old socket and create a new decoder;
  never send video application bytes. Stop interrupts retry waits.
- Invalidate detector results, Nano and BoT state on stream interruption. Frame
  stream epochs and decision generations reject old in-flight work. Invalidate
  pending Dance movement immediately, including when Nano is off. Clear the old
  GUI image and show reconnecting/recovered status; add recovery diagnostics.
- The session 20261015_141708_55237d contains 49 RC ACK timeout pauses, three
  focus pauses and one enable rejection during takeoff; no UI-heartbeat pause.
  There are 39 decoded-frame gaps above 250 ms and a final no-video-bytes timeout.
  Concurrent telemetry delays suggest transport/phone instability, not proven
  Wi-Fi causality. Recovery fixes do not remove upstream network stalls.
- No changes to MSDKRemote, model, local configuration, 250-ms RC ACK limit,
  20-Hz command cadence, motion limits or background-lock safeguards.
- All 472 release tests passed, including delayed neutral ACK during takeoff,
  incomplete enable handshakes, canceled Dance authority, stale in-flight work,
  real H264 decoding after reconnect, bounded retries and hidden Tk recovery UI.
  Validation used local fixtures only; no aircraft/phone connection or flight test.

## Version 13.0.0 / nano-handoff-5 — Source-aligned forward handoff — 2026-10-14

- Reevaluate guarded authority at the command timestamp. A cached accepted
  YOLO decision crossing 250 ms enters the bounded Nano bridge, instead of
  falling through to a zero command. Semantic evidence fields are unchanged.
- Retain at most eight forward anchors for 350 ms. Pair Nano with a compatible
  prior source when YOLO is ahead; a newer stop/identity change invalidates old
  permission, and a newer slowdown caps it. No deadline is renewed by history.
  Track the time of visual application to avoid applying forward decay twice.
- Weak appearance without coherent foreground motion pauses Nano authority
  immediately, while retaining the visual tracker. Missing flow is not positive
  evidence. Good appearance still works on a hovering target without flow.
- Preserve the 350/400/100 ms forward/semantic/visual budgets, 35% forward
  bridge cap, final user axis caps, proximity, border, identity and manual vetoes.
  A new stop or withdrawal of forward permission also revokes a pending bridge.
- Log handoff reasons and selected source times, and profile camera-motion
  preprocessing, features, optical flow and estimation inside BoT-SORT. No
  MSDKRemote, model, local configuration or command-rate changes.
- All 459 release tests passed. Socket fixtures now wait for actual neutral
  receipt/ACK or sample publication, avoiding races with local status flags.
  On 2,414 exactly reproduced recorded decisions, positive raw forward gates
  increased from 700 to 897; all three source-aligned wall-following commands
  near 88 s were blocked. 138 commands lacked an exact replay match and were
  excluded. These are open-loop decisions, not measured flight improvement.
  Three video excerpts also ran the production workers without network access
  or errors: requested YOLO 10 FPS, achieved 9.45–10.07 FPS; Nano averaged
  22.63–27.85 updates/s. Camera-motion processing averaged 22.31–27.00 ms.

## Version 13.0.0 / nano-continuity-4 — Visual continuity and telemetry — 2026-10-14

- In enabled Dance, a strong original YOLO NEO measurement may repair short
  BoT association loss using strict source-aligned Nano evidence. Initial and
  long-loss acquisition remain unchanged. No Nano box becomes a detection.
- Guarded Nano is the default. A matching delayed anchor preserves a newer
  checked visual frame. Fresh strong YOLO survives ordinary Nano unavailability;
  actual appearance/class/exit conflicts still veto visual control.
- Yaw bridging expires 400 ms after real NEO evidence. Forward bridging only
  decays a previously verified command, capped at 35% of the configured forward
  axis limit and expiring at 350 ms from that measurement. Suspect, clipped,
  close and unconfirmed targets cannot authorize forward bridging.
- Two coherent raw NEO frames can guide search yaw up to 20%, within existing
  heading/arc/deadline limits. Episodes last at most 800 ms with 400 ms cooldown;
  they grant no translation and never extend the overall search deadline.
- Python sockets enable TCP_NODELAY. Control retains its 20 Hz target and
  serialized acknowledgements. Read-only telemetry tries LISTEN/UNLISTEN with
  GET heartbeat/fallback and explicit local-time provenance. No Android edits.
- Added offline tracking/loss/search tests and localhost TCP protocol tests.
  All 437 release tests passed, including bounded forward on a mock control
  connection, live vetoes, forward expiry during ACK wait and GET rate limiting.
  Four real-time video excerpts also passed with 10 FPS requested inference and
  30 FPS video sampling; no translation or Nano following was authorized in the
  marked absent-target windows. Independently bounded search yaw is permitted.
  Hardware latency, aircraft response and closed-loop flight are not measured
  by those tests; an Android acknowledgement is not proof of physical motion.

## Version 13.0.0 / nano-template-3 — Confirmed template refresh and keyboard recovery — 2026-10-14

- Nano's DNN template can refresh from a strong, confirmed YOLO NEO anchor
  when its scale or appearance changes, with source-frame agreement, image
  texture checks and a 750 ms minimum interval. Nano predictions never update
  their own template or renew semantic/control authority. Refresh reasons and
  anchor IoU are logged in `nano_anchor_applied`.
- Repeated polling of a paused keyboard connection no longer alternates its
  status text. Unexpected/extra ACKs still quarantine the old connection;
  explicit Enable (E) now requests a fresh connection and a new neutral/enable
  handshake. Held keys/actions are cleared; Q, stop and focus loss prevent
  movement during reconnect. No automatic reconnect on an ambiguous reply.
- Offline paired replay of three clips from `20261014_124648_1e99ec`, with
  YOLO at 10 FPS and Nano on intermediate camera frames, found visual-box
  availability of 51.0% -> 85.6% (distant), 78.6% -> 73.6% (occlusion), and
  50.6% -> 52.2% (fast/exit). These are coverage, not annotated accuracy;
  asynchronous scheduling differs between runs. Average Nano update cost was
  18.8–20.3 ms, with whole-process CPU around 13.6–15.1% of all logical cores.
  Occlusion, small targets and background drift remain limitations. The
  visual-loss checks and 250/100 ms control deadlines were not relaxed.
- 417 offline regression tests passed, including explicit reconnect, stale
  keys, cancellation during connect, and template-update evidence checks.

## Version 13.0.0 / nano-continuity-2 — Visual continuity and aligned overlays — 2026-10-14

- Separate visual lifetime from the 250 ms semantic / 100 ms control freshness
  deadlines. Expiry blocks guarded movement without deleting the Nano template.
  Continue visual updates as UNVERIFIED; explicit visual loss still resets the
  tracker and requires new YOLO evidence. Shadow remains the default.
- Remove forced half-second reinitialization and stale catch-up work; read the
  latest camera frame after initialization. Log actual Nano update FPS. Treat
  persistently low Nano response as a loss cue, never as semantic proof.
- Separate small flow/appearance control warnings from hard visual loss;
  require coherent foreground flow and a box-size-relative discrepancy for
  flow-based loss. Log the actual rejected box, score and cue measurements.
- Reject persistent loss of trusted appearance (600 ms) or collapse of target
  texture, even with a high Nano score. This can also reject long blur/pose
  changes; reliable long-range dance tracking remains unproven.
- Apply semantic anchors only after source-time agreement checks. Compare MINI
  conflicts with historical Nano positions rather than a newer frame's box.
- Remove segmentation contour drawing, retain masks for processing, and add
  source-aligned blue BoT-SORT and historical orange Nano boxes to Analyzed view.
  Show fresh unverified Nano boxes in live preview/recording, separately from
  movement authority. Refresh recording and GUI from the latest visual result.

## Version 13.0.0 / nano-guard-1 — Separate visual support and final-send veto — 2026-10-13

- Add an independent CPU NanoTrack V2 worker with bounded frame history,
  latest-frame scheduling, at most two catch-up frames, reusable DNN networks
  and source-time alignment (bounded interpolation for skipped source frames).
- Keep YOLO/BoT-SORT measurements, identity, class votes and timestamps separate.
  Only unique strong matched raw NEO detections anchor Nano; ambiguous/weak
  observations cannot refresh its semantic deadline. Default mode is shadow.
- Add guarded yaw assistance, 250 ms semantic / 100 ms visual deadlines,
  geometry/appearance/masked local-flow checks, frame-exit rejection and a
  latched veto requiring two new post-veto strong anchors. Guarded loss blocks
  autonomous COAST and SCAN as well; manual override remains available.
- Check the captured permission and current epoch/deadlines before RC send and
  while ACK is pending; neutralize superseded motion without releasing manual
  authority. Reset smoothing on veto, and prevent old-axis ramp carryover.
- Wake control on visual results, cap OpenCV's native pool at one thread,
  include ACK time in the 20 Hz send period, and draw the preview at fitted
  display resolution while preserving full-resolution snapshots/recordings.
- Fix the final Dance gate to accept all registered, class-map-verified dual
  models already permitted by the GUI (including v6), not only the old 640 model.
- Add isolated network-blocked replay and tests for evidence separation,
  stale/racing results, loss, identity changes, final-send cancellation and UI.
  Guarded mode remains experimental: conservative holds reduce continuity;
  neither tracker score nor these checks guarantee collision avoidance.

## Version 13.0.0 / models-v6-1 — Selectable v6 segmentation models — 2026-10-12

- Register `neo_minipro4_yolo26s_seg_960x544_v6.onnx` and
  `neo_minipro4_yolo26s_seg_best_full_ap_960x544_v6.onnx` with separate verified
  SHA-256 hashes. Both use FP32, fixed 960x544 input and raw segmentation outputs.
- Add both to the existing model selector and widen it to show their filenames.
  Retain one model per connection, persisted selection, existing models and
  preferences, NEO-only tracking policy and class 0's `neo` display alias.
- Preserve all supplied PT/ONNX files and the current preprocessing pipeline.
  Existing session logs identify the selected filename, checksum and class map.
- Validation: 374 regression tests, hidden Tk selection/persistence checks and
  the standard launcher's offline GPU checks passed for both new models.
  Six recorded-video frames per model produced boxes, masks and both class
  labels; DirectML execution was verified. This is an integration smoke check,
  not an accuracy ranking. Model, configuration and dependency hashes were
  unchanged. Evidence: `logs/models_v6_validation/`.

## Version 13.0.0 / model-960-1 — Improved PT at 960x544 FP32 — 2026-10-10

- Export a new fixed-input ONNX from the improved Full HD PT using Ultralytics
  8.4.175, FP32, opset 17 and raw segmentation outputs. Preserve all originals;
  all 236 floating-point initializer tensors match the Full HD export exactly.
- Register `yolo26s_seg_best_mp4_960x544_v2_fp32.onnx` with a SHA-256 checksum
  and the verified NEO/MINI mapping. Add it to the existing model picker without
  changing saved preferences, the default model or NEO-only control behavior.
- Full HD frames become 960x540 plus two padding rows per vertical edge; retain
  boxes and masks in source coordinates. Add contract and coordinate tests.
- Extend offline model comparison with selectable models, sequential decoding,
  frame hashes, repeat timings and confidence-controlled previews.
- Validation: 374 regression tests, hidden Tk selection/persistence, dependency
  checks and the standard launcher's offline DirectML GPU check passed.
- GTX 1050 / DirectML, 69 identical sampled frames from two recordings, three
  passes per model: 960 FP32 14.26 FPS, Full HD 4.41 FPS, previous 640 17.53 FPS.
  Includes preprocessing/inference/masks, excludes live UI/video/recording.
  The smaller export misses several visible small NEOs retained by Full HD;
  the speed gain is not accuracy preservation. No ground-truth mAP was measured.
  Full results and annotated comparisons: `logs/model_960_validation/REPORT_HE.md`.

## Version 13.0.0 / directml-restored-1 — Remove the CUDA experiment — 2026-10-10

- Remove the isolated CUDA environment and dependencies, dedicated launch/setup
  scripts, CUDA runtime loader, backend selector and CUDA-only comparison/tests.
- Restore the regular DirectML/CoreML/CPU path and `config.local.json` settings.
  Preserve the working `.venv`, original models and recorded experiment logs.
- Retain the dual-resolution model selector, delayed detection overlays and all
  existing NEO tracking/control behavior. No NVIDIA driver changes.
- Validation: 372 regression tests, dependency consistency and hidden Tk GUI
  check passed. The regular CMD launcher verified DirectML GPU execution and
  original checksums for both the 640 and Full HD models, without aircraft I/O.

## Version 13.0.0 / cuda-runtime-1 — Isolated Windows CUDA execution — 2026-10-10

Historical experiment; removed by `directml-restored-1` above.

- Add `RUN_FOLLOW_NEO_CUDA.cmd`, `.venv_cuda`, a pinned CUDA dependency list,
  and a separate setup script. Preserve the regular DirectML environment and
  both original FP32 ONNX models. No system CUDA Toolkit or driver changes.
- Load CUDA 12.4/cuDNN 9.1 DLLs from the active environment for GTX 1050/Pascal
  compatibility. Use CUDA device ordinals instead of DXGI adapter IDs, disable
  TF32, verify the selected EP and profile model nodes during initialization.
- Keep native DirectML/CoreML selection unchanged. Explicit CUDA initialization
  failures cannot silently select CPU, including in Auto mode.
- Store CUDA GUI preferences separately in `config.cuda.local.json`, seeded
  from the existing settings with Compute=GPU and automatic CUDA device selection.
- Log backend, dependency versions, device namespace and provider options;
  add an offline same-frame comparison tool and regression coverage for routing,
  fallback rejection and independent configuration.
- Validation: 379 regression tests passed in the original environment; all seven
  CUDA routing/configuration tests also passed in `.venv_cuda`. Both hidden Tk
  checks passed. The setup verified real CUDA execution of the large FP32 model;
  the new CMD launcher also passed its offline 640-model check.
- Same-frame GTX 1050 measurement on 24 frames / 72 cycles per backend: DirectML
  4.46 detector FPS, CUDA 5.07 (+13.6%); inference means 203.4 vs 177.8 ms.
  These exclude live video, GUI, tracking and recording. ORT versions are
  DirectML 1.20.1 and CUDA 1.20.2. Across 44 sampled frames from two recordings,
  all 26 emitted detections matched in class/count; maximum box-coordinate
  difference was 0.000123 pixels and confidence difference below 0.000003.
  This checks backend consistency, not model accuracy or flight performance.

## Version 13.0.0 / delayed-overlay-1 — Visible slow-model detections — 2026-10-10

- Fix analyzed-frame boxes disappearing before slow Full HD inference reached
  the display. Keep detections/masks on their actual source frame, label its age,
  and retain the stale-frame warning instead of deleting its detections.
- Show raw detections in live preview and prediction recordings for at most
  750ms. After 250ms, use dashed class-colored boxes labeled DELAYED with age;
  do not mark them SELECTED. These are measured positions, not projected positions.
- Clear raw boxes on empty results, expiry, future timestamps and source-size
  mismatch. Log raw display provenance and visibility in overlay metadata.
- Preserve tracking projection, NEO-only control gates and flight deadlines.
- Validation: 372 tests and saved-run video replay. Run 20261010_153806_14a112
  had 276 recorded frames carrying detections but zero visible boxes. The fix
  displays 275; the remaining measurement is older than 750ms. All 623 replay
  timestamps were checked against the original recording; originals are intact.

## Version 13.0.0 / dual-resolution-1 — Selectable Full HD model — 2026-10-10

- Add checksum-verified `yolo26s_seg_best_mp4_1920x1080_v2.onnx` alongside the
  existing 640 default. Select one model before connecting; disconnect to switch.
- Read fixed spatial dimensions from ONNX and validate anchor/prototype shapes.
  The new export takes 1920x1088: Full HD frames retain their pixels with four
  padding rows at each edge. Map both boxes and masks back to source coordinates.
- Preserve model dimensions when applying confidence/NMS settings. Display the
  active input size and log the preprocessing transform with observations.
- Validate the new source names (`NEO`, `MINI PRO4`); alias class 0 to `neo` and
  permit the same NEO-only Dance policy for either registered two-class model.
- Add an offline comparison tool using identical video frames. Existing weights,
  motion settings and stale-data limits are preserved.
- Validation: 366 regression tests passed, including rectangular masks/boxes,
  lossless Full HD letterboxing, settings updates and both model policies.
  Offline Tk verified saved selections. Same-frame GPU comparison on GTX 1050:
  15.7 detector FPS at 640 versus 3.6 at 1920x1088 (33 samples each). The new
  model's 275ms median cycle exceeds the ordinary 250ms movement measurement
  age limit; improved detections alone do not establish usable live tracking.

## Version 13.0.0 / neo-recovery-3 — Measured cadence and bounded class memory — 2026-10-05

- Accommodate recorded 187--234ms detection intervals without discarding coherent
  recovery evidence. Keep eight observations for distant returns, in a 1.2s
  window; cap candidate pauses at 1.4s without renewal by successive candidates.
- Measure candidate receipt freshness independently of inference latency, while
  retaining source/video age checks. Preserve alignment history across brief gaps.
- Ramp automatic search yaw increases; retain immediate stops/braking and the
  existing explicit BOOST test. Retry transient heading failures sooner without
  relaxing stale-heading or ACK limits. Account for rendering time in the GUI cadence.
- Permit only short, geometry-bound, weak support for an ambiguous class result
  after three real strong NEO wins. Preserve raw class scores; never acquire from
  MINI or refresh strong evidence from class memory. Limit this support to centering,
  with zero forward/roll and a hard expiry checked again at command dispatch.
- Add pipeline timing, class-support provenance and search-ramp diagnostics.
- Add offline detector/ONNX-variant benchmarks and saved-frame tracking replay.
  Original production weights/providers remain unchanged. Experimental BBOX FP32
  and FP16 copies remain outside the active model registry; 30 inference FPS has
  not been achieved. See UPDATE13_HE.md for measured rates and validation limits.
- Validation: 358 local regression tests passed, covering cadence, class-memory expiry and
  measured-search recovery. Fixed-video replay accepted 581/1611
  measurements versus the original 496; this is not flight or accuracy validation.

## Version 13.0.0 / neo-recovery-2 — Retained neo identity — 2026-10-05

- Bind neo class provenance to the last confirmed measurement and logical
  target. Retain it across empty/rejected frames so the live Dance gate allows
  bounded coast and measured search. Clear it on reset, frame-size changes and
  numerical recovery. Preserve post-enable measurement and stale-data gates.
- Keep MINI display-only: it cannot refresh target identity or measurement time.
  Preserve weights, confidence thresholds, axis limits and ACK deadlines.
- Add ACK send/reply timestamps, late counts, pending ages and a bounded RC P95
  window; expose these in control snapshots, command logs and the GUI.
- Log decoded-frame gaps, byte freshness, receive wait and decode/publication
  timings; show gap counts and maximum gap in the GUI.
- Add positive controller-to-Dance coast/search regressions, reset/provenance
  guards, transport timing tests and a read-only saved-decision verifier.
- Saved-run verification: all 240 previously class-blocked coast/search
  decisions had matching confirmed neo measurement provenance. Restoring it
  permits 220 nonzero decisions; other intents/gates retain zero output.
  This is authorization replay, not physical flight validation.
- Validation: all 341 release-gate tests pass; hidden Tk initialization passes.
  The model checksum is unchanged. No aircraft/phone connections or live tests.

## Version 13.0.0 — Dual-class detection, neo-only tracking — 2026-10-05

- Register checksum-verified `yolo26s_seg_best_mp4_v2.onnx` as the default,
  including migration of pre-v13 saved model selection. Preserve other settings.
- Alias class 0 to `neo`, class 1 to `MINI PRO4`; retain original ONNX names in
  diagnostic metadata. Keep the weight file byte-for-byte unchanged.
- Decode two scores separately from 32 mask coefficients; use class-aware NMS
  and per-class limits so overlapping or numerous MINI detections do not remove neo.
- Filter to neo before BoT-SORT acquisition, weak association and reacquisition.
  MINI detections stay visible but cannot refresh neo measurement time or identity.
  Retain v12 centering, spacing, prediction and measured search behavior.
- Require the verified dual-class policy and a neo target in live GUI Dance
  transport. Old models remain selectable for detection display only.
- Draw named, confidence-labelled boxes for both classes in both preview modes;
  never mark a coincident MINI box as the selected target. Recorded overlays and
  snapshots use the same renderer.
- Log original/display class maps, class counts, eligible/display-only counts,
  selected class, model policy and the measurement behind sent Dance commands.
- Add offline contracts for dual-class decoding, masks, overlapping boxes,
  MINI-only input, neo loss/return, configuration, rendering, session logs and
  the actual control worker over a local socketpair. No aircraft connection.
- Validation: all 327 release-gate tests pass (19 new contracts), hidden Tk
  initialization passes, and the new model passes the offline GPU check on
  GTX 1050 / DirectML with hardware execution verified. No live classification
  or flight validation was performed.

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
