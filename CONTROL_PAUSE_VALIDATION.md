# Control pause revision 6 — 2026-09-30

Version: 11.1 / control-auto-resume-7 (supersedes explicit resume in revision 6).

The 250 ms RC acknowledgement threshold no longer disconnects control or
automatically sends `disable`. It sends a neutral RC and latches movement pause.
Non-RC commands retain their two-second pause threshold. GUI heartbeat/focus loss
and unexpected responses also pause movement while keeping the connection.

Late ACKs are consumed in command order. The original motion ACK never confirms
the separate neutral command. Movement can resume only with no pending replies,
a confirmed neutral, an unambiguous stream and a fresh GUI heartbeat. ACK-delay
and heartbeat-only pauses now resume automatically without E. Q / Esc still
releases control; focus/input failures and rejected commands do not auto-resume.
Old keys require release and a new press. Queued flight actions and Dance history
are cleared before re-enabling. Existing flight-action holds remain in place;
Dance requires new decisions/measurements after recovery. Search tests stay stopped.

Unexpected nonempty replies are logged as raw hex and quarantined. The stream
has no command IDs, so it cannot be safely resynchronized automatically: the
connection stays open and the operator can choose to disconnect/reconnect.
Empty line separators do not count as acknowledgements.

Q / Esc remains an explicit release; Disconnect/close remains explicit teardown.
Actual EOF or socket failure is reported as a transport loss. Sending neutral
without its ACK is not reported as confirmed stopping of the aircraft.

Offline validation:

- `run_update12_tests.py`: release regressions plus local socket-pair cases for
  missing, late, split, coalesced, rejected and extra replies; immediate neutral
  on key release/reversal; automatic resume, explicit release/disconnect; heartbeat and focus
  pauses; local processing errors; retained tracking/search/recording behavior.
- `verify_control_gui.py`: hidden actual Tk initialization with version/revision,
  search controls and offline start rejection; no aircraft connection.

No MSDKRemote, protocol, saved configuration or model changes. No flight test was
performed; these checks exercise simulated sockets and the local GUI only.
