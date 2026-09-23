"""Authorization gate for the existing NEO centering / spacing intent."""
import math
from .edge_search import yaw_override
from .prediction import (COAST_SECONDS, COAST_YAW, COAST_VERTICAL, STABLE_FORWARD_SECONDS,
                         fade, forward_fade)

AXES = ('yaw', 'vertical', 'roll', 'forward')
ZERO = (0., 0., 0., 0.)


def dance_command(decision,now,started):
    if not decision:return ZERO,'Waiting for a new target'
    stamp=decision.get('prediction_time')
    if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):return ZERO,'Invalid decision time'
    if stamp<=started or not 0<=now-stamp<=.20:return ZERO,'Waiting for a fresh decision'
    state=decision.get('state');phase=decision.get('spacing_phase')
    if decision.get('stale') or state in ('ERROR','WAIT_VIDEO','ABORT_HOVER'):
        return ZERO,'Waiting for current video / detection'
    if phase in ('STOPPED','SEQUENCE_DONE'):return ZERO,'Sequence stopped: explicit one-shot mode'
    turn=yaw_override(decision,now,started)
    if turn:return (turn,0.,0.,0.),'Search / '+('RIGHT' if turn>0 else 'LEFT')+' / yaw 100%'
    if state=='DIRECTIONAL_SEARCH':return ZERO,'Search held: heading, angle or time contract expired'
    if state in ('HOVER_WAIT','WAIT_TARGET','SEARCH'):
        return ZERO,'Hover / '+str(decision.get('reason','waiting_target'))
    age=decision.get('measurement_age_ms')
    if not isinstance(age,(int,float)) or not math.isfinite(age) or age<0:return ZERO,'Waiting for confirmed target'
    send_age=age/1000+now-stamp
    coast=(decision.get('prediction_recovery') is True and state=='COAST'
           and decision.get('confirmed') is True and not decision.get('accepted'))
    if coast:
        if send_age>COAST_SECONDS:return ZERO,'Hover / prediction expired'
    elif (send_age>.25 or state not in ('TRACK','REACQUIRE')
          or not decision.get('accepted') or not decision.get('confirmed')):
        return ZERO,'Waiting for confirmed target'
    try:
        policy_values=tuple(float(decision['intent'][axis]) for axis in AXES)
        basis=decision.get('intent_basis') or {axis:1. for axis in AXES}
        scales=tuple(float(basis[axis]) for axis in AXES)
    except (KeyError,ValueError,TypeError):return ZERO,'Invalid tracking intent'
    if (not all(math.isfinite(v) and -1<=v<=1 for v in policy_values)
        or not all(math.isfinite(v) and 0<v<=1 for v in scales)):
        return ZERO,'Invalid tracking intent'
    values=tuple(max(-1.,min(1.,v/b)) for v,b in zip(policy_values,scales))
    if coast:
        initial=fade(age/1000);extra=fade(send_age)/initial if initial else 0.
        yaw=max(-COAST_YAW,min(COAST_YAW,values[0]))*extra
        vertical=max(-COAST_VERTICAL,min(COAST_VERTICAL,values[1]))*extra
        forward=0.
        if (decision.get('prediction_forward_allowed') is True
            and decision.get('prediction_quality',{}).get('stable') is True
            and not decision.get('spacing_close_guard_armed') and not decision.get('bbox_clipped')
            and phase=='APPROACH' and send_age<STABLE_FORWARD_SECONDS):
            initial=forward_fade(age/1000)
            forward=min(forward_fade(send_age),max(0.,values[3])*forward_fade(send_age)/initial) if initial else 0.
        return (yaw,vertical,0.,forward),'Kalman recovery / '+str(phase)
    if state=='REACQUIRE' or phase in ('HOLD_REACQUIRE','TRACK_PAUSE','CONFIRM_STOP'):
        values=(values[0],values[1],0.,0.)
    return values,'Tracking / '+str(phase)+(' / REACQUIRE' if state=='REACQUIRE' else '')


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
