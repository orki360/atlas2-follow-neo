"""NEO Dance centering and deterministic recovery logic, perception-only port."""
import math
from .types import Intent, clamp


def interpolate(x,a,b,c,d): return c+clamp((x-a)/(b-a),0,1)*(d-c)


class SmartTrackingController:
    def __init__(self):
        self.axes = [{'centered':True,'last':0,'pending':0,'since':None} for _ in range(2)]
        self.roll = 0.; self.roll_time = None; self.standoff = False

    def axis(self,error,yaw,now,limit):
        s = self.axes[0 if yaw else 1]; distance = abs(error)
        if s['centered'] and distance<=.075 or not s['centered'] and distance<=.035:
            s.update(centered=True,pending=0,since=None); return 0.
        s['centered'] = False
        low,near,mid,maximum = (.08,.16,.38,.70) if yaw else (.07,.14,.30,.55)
        speed = (interpolate(distance,.035,.20,low,near) if distance<=.20 else
                 interpolate(distance,.20,.45,near,mid) if distance<=.45 else
                 interpolate(distance,.45,1,mid,maximum))
        speed = clamp(speed*limit/maximum*(1.05 if yaw else 1),0,limit)
        command = math.copysign(speed,error)*(1 if yaw else -1)
        direction = 1 if command>0 else -1
        if s['last']==0 or direction==s['last']:
            s.update(last=direction,pending=0,since=None); return command
        if s['pending']!=direction or s['since'] is None:
            s.update(pending=direction,since=now); return 0.
        if now-s['since']<.18: return 0.
        s.update(last=direction,pending=0,since=None); return command

    def compute(self,k,w,h,age,occluded,now,settings):
        if not k.initialized: return Intent(),0.,0.
        ex = (k.cx-w*.5)/(w*.5); ey = (k.cy-h*.5)/(h*.5)
        yaw = self.axis(ex,True,now,settings.yaw_limit)
        vertical = self.axis(ey,False,now,settings.vertical_limit)
        ratio = k.width/w
        self.standoff = ratio>=.80 or self.standoff and ratio>=.70
        forward = 0. if self.standoff or occluded or age>.65 or abs(ex)>.85 or abs(ey)>.80 else (1. if ratio<=.68 else interpolate(ratio,.68,.80,1,.60))
        requested = 0.
        if occluded or age>.55:
            self.roll=0.; self.roll_time=now
        elif abs(ex)<=.65 and abs(k.vx/w)>.06:
            requested = math.copysign(interpolate(abs(k.vx/w),.06,.65,.06,.45),k.vx)*max(.25,1-abs(ex)/.65)
        dt = 0 if self.roll_time is None else clamp(now-self.roll_time,0,.20)
        self.roll += clamp(requested-self.roll,-1.2*dt,1.2*dt); self.roll_time=now
        return Intent(yaw,vertical,self.roll,forward),ex,ey


class DeterministicTrackingPolicy:
    """Recoverable target states; only real measurements can confirm a track."""
    def __init__(self):
        self.state='WAIT_TARGET';self.entered=0.;self.reason='waiting_first_detection'
        self.edge='';self.current_edge='';self.last_id=None;self.hits=0;self.ever=False
        self.last=Intent();self.last_measurement_time=None;self.reacquire_since=None
        self.transition_count=0;self.transition=None
        self.measurement_fresh=False;self.new_measurement=False;self.time_since_detection=math.inf

    def transition_to(self,state,reason,now):
        self.transition=None
        if state!=self.state:
            self.transition_count+=1
            self.transition=dict(previous=self.state,current=state,reason=reason,
                previous_state_age_ms=max(0.,now-self.entered)*1000.,transition_count=self.transition_count)
            self.state=state;self.entered=now
        self.reason=reason

    def update(self,now,k,w,h,measurement_time,measurement_id,command,settings,abort=False,
               accepted=True,search=None):
        previous_time=self.last_measurement_time
        if measurement_time is not None and math.isfinite(measurement_time) and measurement_time<=now:
            if previous_time is None or measurement_time>previous_time:self.last_measurement_time=measurement_time
        age=math.inf if self.last_measurement_time is None else max(0.,now-self.last_measurement_time)
        fresh=accepted and age<=.25 and measurement_id is not None
        new=fresh and measurement_id!=self.last_id
        self.time_since_detection=age;self.measurement_fresh=fresh;self.new_measurement=new
        if new:self.last_id=measurement_id
        self.current_edge=''
        if search and search.get('candidate_direction'):
            self.current_edge='right' if search['candidate_direction']>0 else 'left'
            self.edge=self.current_edge
        if abort:
            state,reason='WAIT_VIDEO','video_or_result_unavailable';self.hits=0;self.reacquire_since=None
        elif fresh:
            self.ever=True
            short_return=(self.state=='COAST' and previous_time is not None
                          and measurement_time-previous_time<=.25)
            if self.state=='TRACK' or short_return:
                state,reason='TRACK','fresh_detection';self.hits=max(3,self.hits)
            else:
                if self.state!='REACQUIRE':self.hits=0;self.reacquire_since=None
                if new:
                    self.hits+=1
                    if self.reacquire_since is None:self.reacquire_since=measurement_time
                ready=(self.hits>=3 and self.reacquire_since is not None
                       and measurement_time-self.reacquire_since>=.06)
                state='TRACK' if ready else 'REACQUIRE'
                reason='reacquire_confirmed' if ready else 'reacquire_confirming'
            if state=='TRACK':self.last=command.bounded()
        elif search and search.get('active'):
            state,reason='DIRECTIONAL_SEARCH','directional_search';self.hits=0
        elif search and search.get('consumed'):
            state,reason='HOVER_WAIT',search['reason'];self.hits=0
        elif not self.ever:
            state,reason='WAIT_TARGET','waiting_first_detection';self.hits=0
        elif age<=.65:
            state,reason='COAST','short_detection_gap';self.hits=0
        else:
            state='HOVER_WAIT';reason=(search or {}).get('reason','target_lost')
            if age>=settings.edge_search_seconds:reason='search_timeout'
            self.hits=0
        self.transition_to(state,reason,now)
        if state=='TRACK':return command.bounded()
        if state=='REACQUIRE':return Intent(clamp(command.yaw,-.25,.25),clamp(command.vertical,-.20,.20))
        if state=='DIRECTIONAL_SEARCH':return Intent(yaw=float(search['direction']))
        return Intent()
