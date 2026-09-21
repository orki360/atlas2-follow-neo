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
    def __init__(self):
        self.state='SEARCH'; self.entered=0.; self.reason='waiting_first_detection'
        self.edge=''; self.last_id=None; self.hits=0; self.ever=False; self.last=Intent()
        self.last_measurement_time=None

    def update(self,now,k,w,h,measurement_time,measurement_id,command,settings,abort=False):
        if measurement_time is not None and measurement_time>0 and (self.last_measurement_time is None or measurement_time>self.last_measurement_time):
            self.last_measurement_time=measurement_time
        age = math.inf if self.last_measurement_time is None else max(0,now-self.last_measurement_time)
        fresh = age<=.30; new = fresh and measurement_id is not None and measurement_id!=self.last_id
        if new: self.last_id=measurement_id
        edge=[]
        if k.initialized:
            if k.cx+k.width/2>=w*.96 and (k.vx/w>=.03 or k.cx>=w): edge.append('right')
            elif k.cx-k.width/2<=w*.04 and (k.vx/w<=-.03 or k.cx<=0): edge.append('left')
            if k.cy+k.height/2>=h*.96 and (k.vy/h>=.03 or k.cy>=h): edge.append('bottom')
            elif k.cy-k.height/2<=h*.04 and (k.vy/h<=-.03 or k.cy<=0): edge.append('top')
        if edge: self.edge='+'.join(edge)
        if abort:
            state,reason='ABORT_HOVER','stale_video_or_result'; self.hits=0
        elif fresh:
            self.ever=True; self.last=command.bounded()
            if self.state=='TRACK': state,reason='TRACK','fresh_detection'
            else:
                self.hits=self.hits+int(new) if self.state=='REACQUIRE' else int(new)
                state='TRACK' if self.hits>=3 else 'REACQUIRE'
                reason='reacquire_confirmed' if self.hits>=3 else 'reacquire_confirming'
        elif not self.ever:
            state,reason='SEARCH','waiting_first_detection'; self.hits=0
        else:
            self.hits=0
            if edge and age<=1.60: state,reason='EDGE_RECOVERY','predicted_edge_exit'
            elif age<=.65: state,reason='COAST','short_detection_gap'
            elif age<=settings.search_timeout: state,reason='SEARCH','recovery_timeout_search'
            else: state,reason='ABORT_HOVER','target_lost_timeout'
        if state!=self.state: self.entered=now
        self.state=state; self.reason=reason
        out=Intent()
        if state=='TRACK': out=command
        elif state=='REACQUIRE': out=Intent(clamp(command.yaw,-.25,.25),clamp(command.vertical,-.20,.20))
        elif state=='COAST':
            scale=1-.65*clamp((age-.30)/.35,0,1)
            out=Intent(self.last.yaw*scale,self.last.vertical*scale,self.last.roll*scale*.55,min(max(0,self.last.forward),.25)*scale)
        elif state=='EDGE_RECOVERY':
            decay=max(.35,1-.65*(now-self.entered)/1.6)
            roll=.18 if 'right' in self.edge else -.18 if 'left' in self.edge else 0
            out=Intent(clamp(.70*(k.cx-w/2)/(w/2),-.65,.65)*decay,
                       clamp(-.55*(k.cy-h/2)/(h/2),-.40,.40)*decay,roll*decay,0)
        elif state=='SEARCH' and self.ever:
            direction=1 if 'right' in self.edge else -1 if 'left' in self.edge else math.copysign(1,self.last.yaw or 1)
            phase=int(max(0,now-self.entered)/.65)
            magnitude=settings.search_yaw*(.65+.35*clamp((phase//2)/3,0,1))
            out=Intent(direction*(1 if phase%2==0 else -1)*magnitude)
        return out.bounded()
