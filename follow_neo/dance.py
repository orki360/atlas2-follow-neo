"""Authorization gate for the existing NEO centering / spacing intent."""
import math

AXES = ('yaw', 'vertical', 'roll', 'forward')
ZERO = (0., 0., 0., 0.)


def dance_command(decision, now, started):
    if not decision:
        return ZERO, 'Waiting for a new target'
    stamp = decision.get('prediction_time')
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        return ZERO, 'Invalid decision time'
    if stamp <= started or not 0 <= now - stamp <= .20:
        return ZERO, 'Waiting for a fresh decision'
    if decision.get('stale') or decision.get('state') == 'ERROR':
        return ZERO, 'Video / detection unavailable'
    if decision.get('spacing_phase') in ('STOPPED', 'SEQUENCE_DONE'):
        return ZERO, 'Sequence stopped: restart Dance to reset'
    age = decision.get('measurement_age_ms')
    if (decision.get('state') != 'TRACK' or not decision.get('confirmed')
            or not decision.get('accepted') or not isinstance(age, (float, int))
            or not math.isfinite(age) or not 0 <= age
            or age + (now - stamp) * 1000 > 250):
        return ZERO, 'Waiting for confirmed target'
    try:
        values = tuple(float(decision['intent'][axis]) for axis in AXES)
    except (KeyError, TypeError, ValueError):
        return ZERO, 'Invalid tracking intent'
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in values):
        return ZERO, 'Invalid tracking intent'
    return values, 'Tracking / ' + decision.get('spacing_phase', '')
