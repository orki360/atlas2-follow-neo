"""Authorization gate for the existing NEO centering / spacing intent."""
import math
from follow_neo.control.nano_guard import guard_reason, FORWARD_BRIDGE_SECONDS, MEASUREMENT_SECONDS, current_visual_support
from follow_neo.control.edge_search import yaw_override
from follow_neo.control.recovery_search import scan_command
from follow_neo.control.pursuit import recovery_vector
from follow_neo.control.curve import verified_curve
from follow_neo.control.control_status import target_status_text
from follow_neo.perception.models import DUAL_MODELS,NEO_CLASS_ID,NEO_CLASS_NAME
from follow_neo.perception.prediction import (COAST_SECONDS, COAST_YAW, COAST_VERTICAL, STABLE_FORWARD_SECONDS,
                         fade, forward_fade)

AXES = ('yaw', 'vertical', 'roll', 'forward')
ZERO = (0., 0., 0., 0.)


def dance_command(decision,now,started,*,require_target_policy=False,resume_after=None):
    if not decision:return ZERO,'Waiting for a new target'
    visual=decision.get('nano')
    independent=decision.get('nano_authority') in ('yolo_only','search','recovery_vector')
    visual_block=None if independent else guard_reason(visual,now)
    if visual_block or decision.get('nano_authority')=='blocked':
        return ZERO,'Nano hold / '+str(visual_block or decision.get('nano_block_reason'))
    if visual and visual.get('mode')=='guarded' and not independent:
        if (visual.get('identity')!=(decision.get('target_identity') or {}).get('logical_target_id')
                or decision.get('nano_authority') not in ('bridge_yaw','bridge_forward','verified_yaw','curve_bridge')):
            return ZERO,'Nano hold / identity or authority missing'
    # Live GUI transport requires this contract. The optional argument retains
    # the historical headless replay API; a supplied policy is always enforced.
    policy=decision.get('target_policy')
    if policy is not None or require_target_policy:
        if not (isinstance(policy,dict) and policy.get('dance_allowed') is True
                and policy.get('model_name') in DUAL_MODELS
                and policy.get('target_class_id')==NEO_CLASS_ID
                and policy.get('target_class_name')==NEO_CLASS_NAME):
            return ZERO,'Dance blocked: select the dual-class neo model'
        if (decision.get('selected_class_id')!=NEO_CLASS_ID
                or decision.get('selected_class_name')!=NEO_CLASS_NAME):
            return ZERO,'Waiting for confirmed neo target'
    stamp=decision.get('prediction_time')
    if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):return ZERO,'Invalid decision time'
    if stamp<=started or not 0<=now-stamp<=.20:return ZERO,'Waiting for a fresh decision'
    if resume_after is not None:
        video_time=decision.get('video_time')
        if (not isinstance(video_time,(int,float)) or not math.isfinite(video_time)
                or not resume_after<video_time<=now or stamp<=resume_after):
            return ZERO,'Recovery hold / waiting for new video and decision'
        search=decision.get('edge_search') or {}
        recovering_search=bool(decision.get('state')=='DIRECTIONAL_SEARCH'
            and search.get('active') and (yaw_override(decision,now,started) or scan_command(decision,now,started)))
        if not recovering_search:
            source=decision.get('measurement_time')
            if not isinstance(source,(int,float)) or not math.isfinite(source) or source<=resume_after:
                return ZERO,'Recovery hold / waiting for new target measurement'
            if decision.get('nano_authority') in ('bridge_yaw','bridge_forward') and visual.get('semantic_time',0)<=resume_after:
                return ZERO,'Recovery hold / waiting for new Nano anchor'
    if decision.get('nano_authority')=='curve_bridge':
        curve=verified_curve(decision,visual,now,max(started,resume_after or started))
        if not curve['allowed']:return ZERO,'Curve hold / '+curve['reason']
        return (curve['yaw'],0.,0.,curve['forward']),'Verified curve / bounded decay'
    vector=recovery_vector(decision,visual,now,max(started,resume_after or started))
    if decision.get('nano_authority')=='recovery_vector':
        if not vector['allowed']:return ZERO,'Recovery vector hold / '+vector['reason']
        basis=(decision.get('intent_basis') or {}).get('yaw',0.)
        yaw=(decision.get('intent') or {}).get('yaw',0.)
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (basis,yaw)) or basis<=0:
            return ZERO,'Recovery vector hold / invalid yaw'
        return (max(-.25,min(.25,yaw/basis)),0.,vector['roll'],vector['forward']),'Recovery vector / bounded translation'
    if decision.get('nano_authority') in ('bridge_yaw','bridge_forward'):
        if (decision.get('stale') or visual.get('semantic_time',0.)<=started
                or not decision.get('confirmed') or decision.get('selected_class_id')!=NEO_CLASS_ID
                or decision.get('state') not in ('TRACK','COAST','REACQUIRE')
                or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE','BRAKE','CONFIRM_STOP')):
            return ZERO,'Nano hold / waiting verified identity'
        basis=decision.get('intent_basis') or {};intent=decision.get('intent') or {}
        yaw_basis=basis.get('yaw',0.);value=intent.get('yaw',0.)
        if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (yaw_basis,value)) or yaw_basis<=0:
            return ZERO,'Nano hold / invalid yaw'
        forward=0.;source=decision.get('nano_forward_source_time')
        if decision.get('nano_authority')=='bridge_forward':
            forward_basis=basis.get('forward');forward_value=intent.get('forward')
            if (not all(isinstance(v,(float,int)) and math.isfinite(v) for v in (forward_basis,forward_value))
                    or not 0<forward_basis<=1 or not 0<=forward_value<=1):
                return ZERO,'Nano hold / invalid forward'
        if (decision.get('nano_authority')=='bridge_forward' and isinstance(source,(float,int))
                and started<source<=now and now-source<FORWARD_BRIDGE_SECONDS
                and decision.get('spacing_phase')=='APPROACH' and not decision.get('spacing_close_guard_armed')
                and not decision.get('bbox_clipped') and basis.get('forward',0)>0):
            applied=decision.get('nano_intent_time',stamp)
            if not isinstance(applied,(int,float)) or not math.isfinite(applied) or not stamp<=applied<=now:
                return ZERO,'Nano hold / invalid application time'
            initial=max(0.,min(1.,(FORWARD_BRIDGE_SECONDS-(applied-source))/.15))
            decay=max(0.,min(1.,(FORWARD_BRIDGE_SECONDS-(now-source))/.15))
            if initial:forward=max(0.,min(.35,intent.get('forward',0.)/basis['forward']))*decay/initial
        return (max(-.25,min(.25,value/yaw_basis)),0.,0.,forward),'Nano bridge / '+('bounded forward' if forward else 'yaw only')
    if decision.get('state')=='SEARCH_PAUSED':
        return ZERO,'Search paused / '+str(decision.get('reason','verifying'))
    guided_search=bool(((decision.get('edge_search') or {}).get('candidate_guidance')
        or (decision.get('edge_search') or {}).get('direction_memory'))
        and scan_command(decision,now,started))
    if decision.get('tracking',{}).get('acquisition_state') in ('verifying','reacquiring','waiting') and not guided_search:
        return ZERO,'Hover / '+target_status_text(decision)
    measurement_time=decision.get('measurement_time')
    if measurement_time is not None and measurement_time<=started:
        return ZERO,'Waiting for a measurement after Dance started'
    class_support=decision.get('class_support')
    if class_support:
        real_search=(decision.get('state')=='DIRECTIONAL_SEARCH'
            and isinstance((decision.get('edge_search') or {}).get('source_time'),(float,int))
            and started<class_support.get('last_real_neo_time',0.)
            and (decision.get('edge_search') or {})['source_time']<=class_support.get('last_real_neo_time',0.))
        if (class_support.get('last_strong_neo_time',0.)<=started
            or now>class_support.get('expires_at',0.) and not real_search):
            return ZERO,'Waiting for renewed neo class evidence'
    state=decision.get('state');phase=decision.get('spacing_phase')
    if class_support and state=='DIRECTIONAL_SEARCH' and not real_search:
        return ZERO,'Class evidence required for search'
    if decision.get('stale') or state in ('ERROR','WAIT_VIDEO','ABORT_HOVER'):
        return ZERO,'Waiting for current video / detection'
    if phase in ('STOPPED','SEQUENCE_DONE'):return ZERO,'Sequence stopped: explicit one-shot mode'
    if decision.get('candidate_pending') and not decision.get('accepted') and not guided_search:
        return ZERO,'Hover / verifying YOLO candidate'
    turn=yaw_override(decision,now,started)
    if turn:return (turn,0.,vector['roll'],vector['forward']),'Search / '+('RIGHT' if turn>0 else 'LEFT')+' / yaw 100%'
    search=decision.get('edge_search') or {}
    if state=='DIRECTIONAL_SEARCH' and search.get('phase')=='SCAN':
        turn=scan_command(decision,now,started)
        return (turn,0.,vector['roll'] if turn else 0.,vector['forward'] if turn else 0.),'Scan / '+('RIGHT' if turn>0 else 'LEFT' if turn<0 else 'PAUSED')+(' / yaw up to 100%, braking' if search.get('yaw_full_scale') else ' / normal yaw limit')
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
    elif (send_age>MEASUREMENT_SECONDS or state not in ('TRACK','REACQUIRE')
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
    if class_support:
        values=(max(-COAST_YAW,min(COAST_YAW,values[0])),
                max(-COAST_VERTICAL,min(COAST_VERTICAL,values[1])),0.,0.)
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
    if (values[2] and send_age>.15 and (decision.get('centering_control') or {}).get('lateral_control')
            and not current_visual_support(visual,now,
                (decision.get('target_identity') or {}).get('logical_target_id'),decision.get('generation'),
                decision.get('track_box'))):
        values=(values[0],values[1],0.,values[3])
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

    def update(self,values,now,active=True,damped_yaw=False,damped_lateral=False):
        if not active or not math.isfinite(now):
            self.reset(); return ZERO
        if self.last_time is None or not 0<=now-self.last_time<=.25:
            self.values=ZERO; dt=.05
        else: dt=now-self.last_time
        result=[]
        for i,(old,target,rate) in enumerate(zip(self.values,values,self.RATES)):
            if not math.isfinite(target): self.reset(); return ZERO
            target=max(-1.,min(1.,target))
            if i==0 and damped_yaw and (target==0 or old*target<0 or abs(target)<abs(old)):
                # Let early braking take effect promptly; a reversal still passes zero.
                value=0. if target==0 or old*target<0 else target
            elif (i==3 or i==2 and damped_lateral) and (target==0 or old*target<0 or abs(target)<abs(old)):
                value=0. if old*target<0 else target
            else:
                waypoint=0. if old*target<0 else target
                value=old+max(-rate*dt,min(rate*dt,waypoint-old))
            result.append(value)
        self.values=tuple(result); self.last_time=now
        return self.values


class SearchSmoother:
    """Bound search yaw acceleration; every stop/brake still takes effect now."""
    RISE_PER_SECOND = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.value=0.;self.time=None

    def update(self,values,now,active):
        target=float(values[0]) if active else 0.
        if not math.isfinite(now) or not math.isfinite(target):
            self.value=0.;self.time=None;return ZERO
        dt=.05 if self.time is None else max(0.,min(.20,now-self.time))
        if self.time is not None and not 0<=now-self.time<=.25:self.value=0.;dt=.05
        target=max(-1.,min(1.,target))
        if not active or target==0 or target*self.value<0:
            self.value=0.
        elif abs(target)<abs(self.value):self.value=target
        else:self.value=math.copysign(min(abs(target),abs(self.value)+self.RISE_PER_SECOND*dt),target)
        self.time=now
        return (self.value,0.,0.,0.)
