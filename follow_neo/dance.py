"""Authorization gate for the existing NEO centering / spacing intent."""
import math
from .edge_search import yaw_override
from .prediction import (COAST_SECONDS, COAST_YAW, COAST_VERTICAL, STABLE_FORWARD_SECONDS,
                         fade, forward_fade)

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
    if decision.get('spacing_phase') == 'TRACK_PAUSE':
        return ZERO, 'Reconfirming target: movement held at zero'
    if decision.get('spacing_phase') == 'CONFIRM_STOP':
        return ZERO, 'Checking BBOX threshold: movement held at zero'
    turn=yaw_override(decision,now,started)
    if turn:
        return (turn,0.,0.,0.), 'Edge search / ' + ('RIGHT' if turn>0 else 'LEFT') + ' / yaw 100%'
    age = decision.get('measurement_age_ms')
    bridging=(decision.get('brief_detection_gap') is True
              and not decision.get('spacing_close_guard_armed')
              and not decision.get('bbox_clipped')
              and decision.get('spacing_phase')=='APPROACH')
    coast=(decision.get('prediction_recovery') is True
           and decision.get('state') in ('TRACK','COAST','EDGE_RECOVERY')
           and decision.get('confirmed') and not decision.get('accepted')
           and not decision.get('spacing_close_guard_armed') and not decision.get('bbox_clipped')
           and decision.get('spacing_phase')=='APPROACH')
    if (not isinstance(age,(float,int)) or not math.isfinite(age) or age<0
            or age+(now-stamp)*1000>(COAST_SECONDS*1000 if coast else 250)
            or not coast and (decision.get('state')!='TRACK' or not decision.get('confirmed')
                              or not (decision.get('accepted') or bridging))):
        return ZERO, 'Waiting for confirmed target'
    try:
        policy_values = tuple(float(decision['intent'][axis]) for axis in AXES)
        basis = decision.get('intent_basis') or {axis:1. for axis in AXES}
        basis_values = tuple(float(basis[axis]) for axis in AXES)
    except (KeyError, TypeError, ValueError):
        return ZERO, 'Invalid tracking intent'
    if (not all(math.isfinite(value) and -1 <= value <= 1 for value in policy_values)
            or not all(math.isfinite(value) and 0 < value <= 1 for value in basis_values)):
        return ZERO, 'Invalid tracking intent'
    values=tuple(max(-1.,min(1.,value/scale))
                 for value,scale in zip(policy_values,basis_values))
    if coast:
        age_send=age/1000+now-stamp
        initial=fade(age/1000); extra=fade(age_send)/initial if initial else 0.
        yaw=max(-COAST_YAW,min(COAST_YAW,values[0]))*extra
        vertical=max(-COAST_VERTICAL,min(COAST_VERTICAL,values[1]))*extra
        # Roll expires at 250 ms. Forward can use the separate consistent-
        # trajectory contract, with an independent send-time deadline/decay.
        translate=fade(age_send,.25)/fade(age/1000,.25) if bridging and fade(age/1000,.25) else 0.
        forward=values[3]*translate
        if (decision.get('prediction_forward_allowed') is True
                and decision.get('prediction_quality',{}).get('stable') is True):
            scale=forward_fade(age/1000)
            forward=(max(0.,values[3])*forward_fade(age_send)/scale if scale else 0.)
            # Also bound a malformed intent by the absolute recovery envelope.
            forward=min(forward,forward_fade(age_send))
        if age_send>=STABLE_FORWARD_SECONDS: forward=0.
        return (yaw,vertical,values[2]*translate,forward), 'Kalman recovery / APPROACH'
    if bridging:
        # Continue the decay at transport time as the same decision ages.
        at_decision=max(0.,min(1.,(250-age)/130))
        at_send=max(0.,min(1.,(250-age-(now-stamp)*1000)/130))
        values=tuple(v*at_send/at_decision if at_decision else 0. for v in values)
    return values, 'Tracking / ' + decision.get('spacing_phase', '') + (' / brief gap' if bridging else '')


class DanceSmoother:
    """Time-based normalized slew, separate from final hardware axis caps.

    No minimum-command boost. Stops, invalid input and lost authority bypass
    the ramp. Forward reduction also bypasses it to preserve BBOX deceleration.
    """
    RATES=(1.5,1.5,2.,3.)

    def __init__(self): self.reset()

    def reset(self):
        self.values=ZERO; self.last_time=None

    def update(self,values,now,active=True):
        if not active or not math.isfinite(now):
            self.reset(); return ZERO
        if self.last_time is None or not 0<=now-self.last_time<=.25:
            self.values=ZERO; dt=.05
        else: dt=now-self.last_time
        result=[]
        for i,(old,target,rate) in enumerate(zip(self.values,values,self.RATES)):
            if not math.isfinite(target): self.reset(); return ZERO
            target=max(-1.,min(1.,target))
            if i==3 and (target==0 or old*target<0 or abs(target)<abs(old)):
                value=0. if old*target<0 else target
            else:
                waypoint=0. if old*target<0 else target
                value=old+max(-rate*dt,min(rate*dt,waypoint-old))
            result.append(value)
        self.values=tuple(result); self.last_time=now
        return self.values
