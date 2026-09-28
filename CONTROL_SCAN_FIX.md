# Keyboard recovery and target-loss scanning

## Desktop changes

- Windows key releases are applied immediately, rather than waiting for Tk's
  idle queue. A physical-key check clears a missed release during GUI refresh.
- Enable, reconnect, and release clear old keys and pending key-release callbacks.
- Control acknowledgements are polled in short intervals. Nonzero manual motion
  is interrupted when its held keys change, control is released, or the UI
  heartbeat expires. RC acknowledgements have a 250ms deadline.
- An interrupted/timed-out channel sends a best-effort neutral/disable pair and
  closes. It does not reuse an ambiguous reply stream or report a late `success`
  as proof that the stop arrived. The GUI reports **STOP UNCONFIRMED**.
- Enable requires acknowledged neutral commands before and after the enable
  handshake, before the application reports keyboard control as active.
- Logs now include keyboard changes, command attempts, acknowledgement latency,
  and unconfirmed stop attempts, in addition to acknowledged flight commands.

## Recovery phases

The live BoT-SORT controller uses two distinct recovery phases:

1. **BOOST:** when recent measured motion supports a horizontal exit, a brief
   yaw-only catch-up may use 100% yaw. It ends after at most 350ms or at the
   configured scan boundary, whichever occurs first.
2. **SCAN:** alternate left/right within the configured total heading span.
   A setting of 20 degrees means -10 to +10 degrees about the loss heading.
   Scan uses up to 25% of the normal yaw cap, slows near its target boundary,
   and requests a 150ms neutral pause before reversal. No translation is sent.

A stationary target disappearing can enter SCAN directly. Both phases share
the configured search-time budget. Fresh heading telemetry is mandatory;
stale video/heading, explicit disable, or timeout stops recovery. The first
associated detection pauses recovery; three qualifying observations confirm
reacquisition. Scan never receives the BOOST-only 100% yaw exception.

The heading envelope is a software command constraint, not a guarantee against
physical overshoot. The change has not been tested in flight.

Historical headless controllers using the legacy backend retain the previous
recovery behavior for comparison. Live GUI sessions use the new behavior.

## Remaining receiver-side requirement

Client changes cannot deliver a stop across a broken network or after a client
process/OS failure. The Android receiver needs an independent command-expiry
watchdog, neutralization on disconnect, and neutral initialization when enabling
control. The locally available Android source does not show these protections.
The user confirmed it is the installed app's source and explicitly requested
that MSDKRemote remain unchanged. This update changes only the desktop project.
No original Android source, installed APK, or aircraft firmware is changed.

## Validation

Local socketpairs simulate delayed/missing acknowledgements, key release and
reversal, and ambiguous replies. GUI tests cover missed Windows releases and
reconnection. Search tests cover alternating boundaries, heading wraparound,
timeout, stale inputs, reacquisition, and separation of BOOST/SCAN yaw limits.
No phone connection or flight commands are used in these tests.
