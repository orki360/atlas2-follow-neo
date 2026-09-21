"""Port of NEO Dance's 8-state Kalman tracker, with explicit freshness guards.

Coordinates are decoded-image pixels; timestamps are local monotonic decode
times, NOT camera exposure times. Width/height are never extrapolated in loss.
"""
import math
import cv2
import numpy as np
from .types import Box, Detection, Kinematics, clamp


def iou(a, b):
    intersection = max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    union = a.width*a.height+b.width*b.height-intersection
    return intersection/union if union > 0 else 0.0


class BBoxTracker:
    def __init__(self, frame_size=(640,480)):
        # Scale noise, speed and association thresholds, not the image itself.
        self.scale=np.array([frame_size[0]/640,frame_size[1]/480]*2,np.float64)
        self.reset()

    def reset(self):
        self.kf = cv2.KalmanFilter(8,4,0,cv2.CV_64F)
        self.kf.transitionMatrix = np.eye(8,dtype=np.float64)
        self.kf.measurementMatrix = np.eye(4,8,dtype=np.float64)
        self.kf.statePost = np.zeros((8,1),np.float64)
        self.kf.errorCovPost = np.eye(8,dtype=np.float64)
        self.kf.measurementNoiseCov = np.eye(4,dtype=np.float64)
        self.kf.processNoiseCov = np.zeros((8,8),np.float64)
        self.initialized = self.confirmed = False
        self.hits = 0
        self.last_time = self.measurement_time = self.measurement_id = None
        self.previous_raw = self.previous_time = self.velocity = None
        self.accepted = None

    def predict_to(self, now):
        if not self.initialized or now <= self.last_time + 1e-6: return
        dt = min(now-self.last_time,.12)
        a = np.eye(8,dtype=np.float64); a[0,4] = a[1,5] = dt
        q = np.zeros((8,8),np.float64)
        for p,v in [(0,4),(1,5)]:
            variance=(900*self.scale[p])**2
            q[p,p] = .25*dt**4*variance
            q[p,v] = q[v,p] = .5*dt**3*variance
            q[v,v] = dt**2*variance
        q[2,2] = (45*self.scale[2])**2*dt
        q[3,3] = (45*self.scale[3])**2*dt
        self.kf.transitionMatrix = a; self.kf.processNoiseCov = q
        self.kf.predict()
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        self.last_time = now

    def snapshot(self, now):
        if not self.initialized: return Kinematics()
        self.predict_to(now)
        a = self.kf.statePost[:,0]
        return Kinematics(True,float(a[0]),float(a[1]),max(2,float(a[2])),
                          max(2,float(a[3])),float(a[4]),float(a[5]))

    def select(self, detections, start_confidence):
        valid = [d for d in detections if math.isfinite(d.confidence)
                 and all(math.isfinite(v) for v in vars(d.box).values())
                 and d.box.width>0 and d.box.height>0]
        if not valid: return None
        if not self.initialized:
            best = max(valid,key=lambda d:d.confidence)
            return best if best.confidence >= start_confidence else None
        predicted = self.snapshot(self.last_time).box
        gate = max(100,1.75*max(predicted.width/self.scale[0],predicted.height/self.scale[1]))
        candidates = []
        for d in valid:
            if d.confidence < .25: continue
            distance = math.hypot((d.box.cx-predicted.cx)/self.scale[0],(d.box.cy-predicted.cy)/self.scale[1])
            overlap = iou(d.box,predicted)
            similarity = math.sqrt(min(d.box.width,predicted.width)/max(d.box.width,predicted.width)
                                   * min(d.box.height,predicted.height)/max(d.box.height,predicted.height))
            if distance>gate and overlap<.05: continue
            if similarity<.45 and overlap<.10: continue
            score = .45*d.confidence+.25*overlap+.15*max(0,1-distance/gate)+.15*similarity
            candidates.append((score,d))
        return max(candidates,key=lambda x:x[0])[1] if candidates else None

    def update(self, detections, source_time, now, frame_id, start_confidence=.45):
        self.accepted = None
        # Duplicate or reordered results must not reconfirm or rejuvenate a track.
        if not all(math.isfinite(x) for x in (source_time,now)) or source_time>now: return False
        if self.measurement_id is not None and (frame_id<=self.measurement_id or source_time<=self.measurement_time): return False
        target = self.select(detections,start_confidence)
        if target is None:
            if not self.confirmed: self.hits = 0
            return False
        raw = np.array([target.box.cx,target.box.cy,max(2*self.scale[2],target.box.width),max(2*self.scale[3],target.box.height)],np.float64)
        if self.previous_time is not None:
            dt = source_time-self.previous_time
            if .015 <= dt <= .60:
                limit=np.array([2500,2500,1400,1400])*self.scale
                v = np.clip((raw-self.previous_raw)/dt,-limit,limit)
                self.velocity = v if self.velocity is None else .25*self.velocity+.75*v
        self.previous_raw = raw.copy(); self.previous_time = source_time
        projected = raw.copy()
        if self.velocity is not None:
            age = clamp(now-source_time,0,.18)
            limit=np.maximum(12*self.scale[:2],1.25*raw[2:])
            projected[:2] += np.clip(self.velocity[:2]*age,-limit,limit)
        if not self.initialized:
            state = np.zeros((8,1),np.float64); state[:4,0] = projected
            if self.velocity is not None: state[4:6,0] = self.velocity[:2]
            self.kf.statePost = state
            self.kf.errorCovPost = np.diag(np.array([12.,12.,20.,20.,1200.,1200.,500.,500.])*np.tile(self.scale,2)**2)
            self.initialized = True; self.last_time = now; self.hits = 1
        else:
            self.predict_to(now)
            # correct() must see the current state even for an equal-time update.
            self.kf.statePre = self.kf.statePost.copy()
            self.kf.errorCovPre = self.kf.errorCovPost.copy()
            self.kf.measurementNoiseCov = np.diag(np.array([4.,4.,12.25,12.25])*self.scale**2)/clamp(target.confidence,.25,1)
            self.kf.correct(projected.reshape(4,1))
            state = self.kf.statePost.copy()
            if self.velocity is not None: state[4:6,0] = .15*state[4:6,0]+.85*self.velocity[:2]
            state[6:8,0] = 0
            self.kf.statePost = state
            self.hits = min(2,self.hits+1)
        self.confirmed = self.confirmed or self.hits>=2
        self.measurement_time = source_time; self.measurement_id = frame_id
        self.accepted = target
        return True
