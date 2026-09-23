"""Single-target Kalman filter corrected at source-frame times.

Decode times are local monotonic timestamps, NOT camera exposure times.
Display/control predictions are read-only; neither advances the filter.
"""
from collections import deque
import math
import cv2
import numpy as np
from .types import Kinematics, clamp
from .prediction import COAST_SECONDS, project_center


def iou(a, b):
    intersection=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    union=a.width*a.height+b.width*b.height-intersection
    return intersection/union if union>0 else 0.0


class BBoxTracker:
    def __init__(self, frame_size=(640,480)):
        self.frame_size=frame_size
        self.scale=np.array([frame_size[0]/640,frame_size[1]/480]*2,np.float64)
        self.reset()

    def reset(self):
        self.kf=cv2.KalmanFilter(8,4,0,cv2.CV_64F)
        self.kf.transitionMatrix=np.eye(8,dtype=np.float64)
        self.kf.measurementMatrix=np.eye(4,8,dtype=np.float64)
        self.kf.statePost=np.zeros((8,1),np.float64)
        self.kf.errorCovPost=np.eye(8,dtype=np.float64)
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float64)
        self.kf.processNoiseCov=np.zeros((8,8),np.float64)
        self.initialized=self.confirmed=False
        self.hits=0
        self.last_time=self.measurement_time=self.measurement_id=None
        self.velocity=self.accepted=self.last_strong_time=None
        self.measurement_weak=False
        self.history=deque(maxlen=24)
        self.process_scale=1.;self.innovation_ratio=None
        self.innovations=deque(maxlen=3)

    def anchor(self):
        if not self.initialized: return None
        a=self.kf.statePost[:,0]
        stable=self.quality().get('motion_consistent',False)
        return dict(time=self.last_time,cx=float(a[0]),cy=float(a[1]),
                    width=max(2.,float(a[2])),height=max(2.,float(a[3])),
                    vx=float(a[4]),vy=float(a[5]),
                    damp_after=.28 if stable else .10,decay_tau=.30 if stable else .16,
                    limit_x=float(max(12*self.scale[0],min(self.frame_size[0]*.25,(2. if stable else .8)*float(a[2])))),
                    limit_y=float(max(12*self.scale[1],min(self.frame_size[1]*.20,(1.5 if stable else .8)*float(a[3])))))

    def uncertainty(self,now):
        """Project covariance without correcting it or renewing real evidence."""
        if not self.initialized:return dict(position_std_px=None,process_scale=self.process_scale)
        dt=max(0.,now-self.last_time)
        a=np.eye(8);a[0,4]=a[1,5]=dt
        covariance=a@self.kf.errorCovPost@a.T
        std=[]
        for p in (0,1):
            q=(180*self.scale[p])**2*self.process_scale*dt**3/3
            std.append(float(math.sqrt(max(0.,covariance[p,p]+q))))
        return dict(position_std_px=std,process_scale=self.process_scale,
                    innovation_ratio=self.innovation_ratio,age_seconds=dt,
                    model='adaptive_constant_velocity')

    def snapshot(self, now):
        anchor=self.anchor()
        if anchor is None: return Kinematics()
        cx,cy,vx,vy=project_center(anchor,now)
        return Kinematics(True,cx,cy,anchor['width'],anchor['height'],vx,vy)

    def _predict_measurement(self, source_time):
        # Only a new YOLO result advances Post. Continuous acceleration noise
        # makes the covariance independent of control/GUI polling frequency.
        dt=source_time-self.last_time
        a=np.eye(8,dtype=np.float64);a[0,4]=a[1,5]=dt
        q=np.zeros((8,8),np.float64)
        for p,v in ((0,4),(1,5)):
            variance=(180*self.scale[p])**2*self.process_scale
            q[p,p]=dt**3/3*variance
            q[p,v]=q[v,p]=dt**2/2*variance
            q[v,v]=dt*variance
        q[2,2]=(45*self.scale[2])**2*dt
        q[3,3]=(45*self.scale[3])**2*dt
        self.kf.transitionMatrix=a;self.kf.processNoiseCov=q
        self.kf.predict()
        # Do not inject the damped display forecast into a CV covariance.
        # The filter's mean and covariance must use the same motion model.

    def select(self, detections, start_confidence, source_time=None,
               strong_confidence=.25, continuation_confidence=.25):
        valid=[d for d in detections if math.isfinite(d.confidence)
               and all(math.isfinite(v) for v in vars(d.box).values())
               and d.box.width>0 and d.box.height>0]
        if not valid: return None
        if not self.initialized:
            best=max(valid,key=lambda d:d.confidence)
            return best if best.confidence>=start_confidence else None
        predicted=self.snapshot(self.last_time if source_time is None else source_time).box
        gate=max(100,1.75*max(predicted.width/self.scale[0],predicted.height/self.scale[1]))
        candidates=[]
        for d in valid:
            if d.confidence<continuation_confidence: continue
            distance=math.hypot((d.box.cx-predicted.cx)/self.scale[0],(d.box.cy-predicted.cy)/self.scale[1])
            overlap=iou(d.box,predicted)
            similarity=math.sqrt(min(d.box.width,predicted.width)/max(d.box.width,predicted.width)
                                 *min(d.box.height,predicted.height)/max(d.box.height,predicted.height))
            if distance>gate and overlap<.05: continue
            if similarity<.45 and overlap<.10: continue
            if d.confidence<strong_confidence:
                if (not self.confirmed or source_time is None or self.last_strong_time is None
                        or not 0<=source_time-self.last_strong_time<=COAST_SECONDS
                        or overlap<.20 or similarity<.65 or distance>gate*.5): continue
            score=.45*d.confidence+.25*overlap+.15*max(0,1-distance/gate)+.15*similarity
            candidates.append((score,d))
        return max(candidates,key=lambda x:x[0])[1] if candidates else None

    def update(self, detections, source_time, now, frame_id, start_confidence=.45,
               strong_confidence=.25, continuation_confidence=.25):
        self.accepted=None
        if not all(math.isfinite(x) for x in (source_time,now)) or source_time>now: return False
        if self.measurement_id is not None and (frame_id<=self.measurement_id or source_time<=self.measurement_time): return False
        target=self.select(detections,start_confidence,source_time,strong_confidence,continuation_confidence)
        if target is None:
            if not self.confirmed: self.hits=0
            return False
        raw=np.array([target.box.cx,target.box.cy,max(2*self.scale[2],target.box.width),
                      max(2*self.scale[3],target.box.height)],np.float64)
        expired=self.last_time is not None and source_time-self.last_time>.60
        if not self.initialized or expired:
            state=np.zeros((8,1),np.float64);state[:4,0]=raw
            self.kf.statePost=state
            self.kf.errorCovPost=np.diag(np.array([12.,12.,20.,20.,1200.,1200.,500.,500.])*np.tile(self.scale,2)**2)
            self.initialized=True;self.hits=1;self.velocity=None;self.history.clear()
            self.process_scale=1.;self.innovation_ratio=None;self.innovations.clear()
        else:
            self._predict_measurement(source_time)
            position_std=np.maximum(3*self.scale[:2],.04*raw[2:])
            noise=np.r_[position_std**2,(3.5*self.scale[2:])**2]
            self.kf.measurementNoiseCov=np.diag(noise)/clamp(target.confidence,.25,1)
            innovation=raw[:2]-self.kf.statePre[:2,0]
            variance=np.diag(self.kf.errorCovPre)[:2]+np.diag(self.kf.measurementNoiseCov)[:2]
            self.innovation_ratio=float(np.mean(innovation**2/np.maximum(variance,1.)))
            self.innovations.append(innovation.copy())
            # Alternating detector jitter must not be mistaken for a maneuver.
            coherent=(len(self.innovations)==3 and target.confidence>=strong_confidence
                      and all(float(a@b)>0 for a,b in zip(self.innovations,list(self.innovations)[1:])))
            wanted=clamp(self.innovation_ratio,1.,9.) if coherent and len(self.history)>=3 else 1.
            self.process_scale=.80*self.process_scale+.20*wanted
            self.kf.correct(raw.reshape(4,1))
            state=self.kf.statePost.copy()
            state[4:6,0]=np.clip(state[4:6,0],-np.array(self.frame_size)*.8,np.array(self.frame_size)*.8)
            state[6:8,0]=0.;self.kf.statePost=state
            self.velocity=state[4:8,0].copy()
            self.hits=min(2,self.hits+1)
        self.last_time=source_time
        self.confirmed=self.confirmed or self.hits>=2
        self.measurement_time=source_time;self.measurement_id=frame_id
        self.accepted=target;self.measurement_weak=target.confidence<strong_confidence
        if not self.measurement_weak: self.last_strong_time=source_time
        self.history.append((source_time,raw.copy(),not self.measurement_weak))
        while self.history and source_time-self.history[0][0]>.30: self.history.popleft()
        return True

    def quality(self):
        """Real-measurement consistency; forecasts never add evidence here."""
        out=dict(stable=False,motion_consistent=False,reason='insufficient_history',samples=len(self.history),
                 span_ms=0.,residual_ratio=None)
        if len(self.history)<4: return out
        times=np.array([r[0] for r in self.history]);times-=times[-1]
        out['span_ms']=float(-times[0]*1000)
        if -times[0]<.10: return out
        values=np.array([r[1] for r in self.history])
        slope,intercept=np.linalg.lstsq(np.c_[times,np.ones(len(times))],values[:,:2],rcond=None)[0]
        residual=np.abs(values[:,:2]-(times[:,None]*slope+intercept))
        tolerance=np.maximum(4*self.scale[:2],.12*np.median(values[:,2:],axis=0))
        ratio=float(np.max(np.percentile(residual,90,axis=0)/tolerance))
        latest=float(np.max(residual[-1]/tolerance))
        out['residual_ratio']=ratio
        size_change=np.max(values[:,2:],axis=0)/np.min(values[:,2:],axis=0)
        state=self.kf.statePost[:,0]
        if sum(r[2] for r in self.history)<3 or self.last_strong_time is None or self.last_time-self.last_strong_time>.10:
            out['reason']='weak_measurements'
        elif ratio>1 or latest>1.5 or np.any(size_change>1.6): out['reason']='inconsistent_boxes'
        elif np.any(np.abs(state[4:6])/np.array(self.frame_size)>.35): out['reason']='fast_image_motion'
        elif np.any(np.abs(state[4:6]-slope)>np.maximum(30*self.scale[:2],.10*values[-1,2:]/(-times[0]))):
            out['reason']='changing_direction'
        else:
            out.update(stable=True,reason='consistent_measurements')
            travel=(values[-1,:2]-values[0,:2])/self.scale[:2]
            steps=np.diff(values[:,:2],axis=0)/self.scale[:2]
            substantial=float(np.linalg.norm(travel))>max(8.,.20*float(np.max(values[-1,2:]/self.scale[2:])))
            out['motion_consistent']=bool(substantial and np.mean(steps@travel>0)>=.75)
        return out
