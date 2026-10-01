"""Measured lateral evidence and user-facing approach diagnostics."""
from collections import deque
import math
import statistics


class LateralEvidence:
    def __init__(self):
        self.previous=None;self.steps=deque(maxlen=8);self.command=0.;self.tick=None
        self.diagnostics={'reason':'waiting_measurements','samples':0}

    def update(self,observation,now,allowed,block_reason):
        dt=0. if self.tick is None else max(0.,min(.1,now-self.tick))
        self.tick=now
        reason=block_reason if not allowed else 'waiting_camera_compensation'
        ready=(allowed and observation and observation.get('camera_valid') and
               0<=now-observation['time']<=.15)
        if not ready:
            self.previous=None;self.steps.clear();self.command=0.
            self.diagnostics=dict(reason=reason,samples=0);return 0.
        o=observation;old=self.previous
        if old is not None and o['time']<old['time']:
            self.previous=None;self.steps.clear();old=None
        if old is None or o['time']>old['time']:
            self.previous=o
            if old is not None:
                span=o['time']-old['time'];warp=o['warp'];x,y=old['point']
                predicted=warp[0][0]*x+warp[0][1]*y+warp[0][2]
                travel=(o['point'][0]-predicted)/o['width']
                speed=travel/span if span>0 else 0.
                if (o['frame_id']!=old['frame_id']+1 or not .005<=span<=.15
                    or not math.isfinite(speed) or abs(speed)>1.):
                    self.steps.clear()
                else:self.steps.append((o['time'],span,travel,speed))
            while self.steps and o['time']-self.steps[0][0]>.35:self.steps.popleft()
        span=sum(s[1] for s in self.steps)
        velocity=statistics.median(s[3] for s in self.steps) if self.steps else 0.
        agree=sum(s[3]*velocity>0 for s in self.steps)/max(1,len(self.steps))
        travel=sum(s[2] for s in self.steps)
        confirmed=(len(self.steps)>=4 and span>=.15 and abs(velocity)>=.06
                   and abs(travel)>=.012 and travel*velocity>0 and agree>=.8
                   and all(s[3]*velocity>0 for s in list(self.steps)[-2:]))
        target=math.copysign(min(.10,.30*(abs(velocity)-.04)),velocity) if confirmed else 0.
        # Invalid evidence removes lateral demand immediately; only rises are ramped.
        if target*self.command<0:self.command=0.
        self.command=math.copysign(min(abs(target),abs(self.command)+.35*dt),target) if target else 0.
        self.diagnostics=dict(reason='confirmed_lateral_motion' if confirmed else 'verifying_lateral_motion',
                              samples=len(self.steps),span_seconds=span,relative_vx=velocity,
                              direction_agreement=agree,roll=self.command)
        return self.command


def forward_diagnostics(state,phase,spacing_reason,spacing_scale,alignment,accepted,clipped,forward):
    if state in ('DIRECTIONAL_SEARCH','SEARCH_PAUSED'):reason='Searching for target'
    elif state not in ('TRACK','COAST') or not accepted:reason='Waiting for verified target'
    elif clipped:reason='Target box at image edge'
    elif phase!='APPROACH':reason='Holding distance' if phase=='VISUAL_HOLD' else 'Confirming distance / target'
    elif spacing_scale<.999:reason='Slowing for distance'
    elif alignment<.999:reason='Aligning with target'
    elif forward<=0:reason='Position / measurement gate'
    else:reason='Using configured forward speed'
    return dict(reason=reason,spacing_scale=spacing_scale,alignment_scale=alignment,
                spacing_reason=spacing_reason,forward_intent=forward)


def forward_status_text(decision,cap):
    info=decision.get('forward_control') or {}
    reason=info.get('reason','Waiting for control data')
    authority=decision.get('control_status')
    if authority is not None:
        if not authority.get('connected'):reason='Control disconnected'
        elif not authority.get('active'):reason='Movement paused'
        elif authority.get('mode')!='DANCE':reason='DANCE off'
    return f"Forward cap: {100*cap:.1f}% | {reason}"
