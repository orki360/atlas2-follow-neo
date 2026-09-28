"""One BoT motion estimate; persistent logical identity and measured recovery."""
import math
from collections import deque
from types import SimpleNamespace
import numpy as np
from .types import Box, Detection, Kinematics
from .camera_motion import CameraMotion
from .prediction import project_center

RECOVERY_SECONDS = 3.0

def low_threshold(confidence):
    return min(.10, confidence*.5)

class BoTSORTTracker:
    backend = 'BoT-SORT'

    def __init__(self, frame_size=(640,480)):
        self.frame_size=frame_size
        self.scale=np.array([frame_size[0]/640,frame_size[1]/480]*2,np.float64)
        self.reset()

    def reset(self):
        self.engine=self.target=None
        self.target_id=self.logical_target_id=None
        self.last_result_time=self.last_result_id=None
        self.measurement_time=self.measurement_id=self.last_strong_time=None
        self.accepted=None;self.confirmed=False;self.hits=0;self.measurement_weak=False
        self.last_detection=None;self.acquisition_confidence=.8
        self.history=deque(maxlen=24)
        self.pending=None;self.candidate_pending=False
        self.pause_started=self.last_candidate_time=None
        self.association_reason='waiting_for_acquisition';self.candidate_details=[]
        self.identity_changes=0;self.identity_event=None
        self.detector_count=0

    def _candidate(self,d,now):
        # Gate against the CURRENT BoT estimate, already compensated for camera motion.
        if self.target is None or self.measurement_time is None:return None
        age=now-self.measurement_time
        if age<0:return None
        mean=self.target.mean;box=d.box
        w,h=max(2.,mean[2]),max(2.,mean[3])
        ratios=np.array([box.width/w,box.height/h])
        size_ok=not (np.any(ratios<.45) or np.any(ratios>2.2))
        delta=np.array([box.cx-mean[0],box.cy-mean[1]])
        std=np.sqrt(np.maximum(0,np.diag(self.target.covariance)[:2]))
        gate=np.minimum(np.array(self.frame_size)*.22,np.maximum([w*2.5,h*2.5],std*3))
        distance=float(np.linalg.norm(delta/np.maximum(gate,1)))
        if age<=RECOVERY_SECONDS and distance<=1 and size_ok:
            return distance+.20*float(np.sum(np.abs(np.log(ratios))))
        # After a genuine gap the old image location is not a permanent veto:
        # the single-target application may verify a unique strong returning
        # candidate anywhere in the frame. This requires a longer sequence.
        if age>=.65 and d.confidence>=self.acquisition_confidence and self.last_detection is not None:
            old=self.last_detection.box
            ratio=np.array([box.width/max(old.width,2),box.height/max(old.height,2)])
            if np.all(ratio>=.2) and np.all(ratio<=5.):
                return 2.+.1*float(np.sum(np.abs(np.log(ratio))))
        return None

    def _recover(self,valid,current,source_time,strong_confidence):
        ranked=sorted(((cost,d) for d in valid
                       if (cost:=self._candidate(d,source_time)) is not None),key=lambda p:p[0])
        self.candidate_details=[dict(confidence=d.confidence,cost=cost,
                                     box=list(vars(d.box).values())) for cost,d in ranked[:4]]
        if not ranked:
            self.pending=None;self.association_reason='no_plausible_candidate';return None
        if len(ranked)>1 and ranked[1][0]-ranked[0][0]<.20:
            self.pending=None;self.association_reason='ambiguous_candidates';return None
        cost,det=ranked[0];old=self.pending;relocation=cost>=2.
        if self.pause_started is None or self.last_candidate_time is None or source_time-self.last_candidate_time>.5:
            self.pause_started=source_time
        self.last_candidate_time=source_time
        if old is not None and old['relocation']!=relocation:old=None
        point=np.array([det.box.cx,det.box.cy]);coherent=False;step=None
        if old is not None and 0<source_time-old['time']<=.15:
            warp=self.engine.gmc.last_warp
            previous=warp[:,:2]@old['point']+warp[:,2];step=point-previous
            scale=max(16.,det.box.width,det.box.height)
            coherent=(np.linalg.norm(step)<=scale*1.5 and .55<=det.box.width/old['width']<=1.8)
            if old['step'] is not None and np.linalg.norm(step)>scale*.15:
                coherent=coherent and float(step@old['step'])>=-scale*scale*.1
        if not coherent:old=None
        self.pending=dict(point=point,width=det.box.width,time=source_time,relocation=relocation,
                          since=source_time if old is None else old['since'],
                          hits=1 if old is None else old['hits']+1,
                          strong=int(det.confidence>=strong_confidence)+(0 if old is None else old['strong']),
                          step=step if coherent else None)
        p=self.pending;self.candidate_pending=source_time-self.pause_started<=.35
        self.association_reason='candidate_verifying'
        strong=p['strong']>=3
        needed=8 if relocation else 3 if strong else 4
        span=.25 if relocation else .06 if strong else .10
        if p['hits']<needed or source_time-p['since']<span:return None
        candidate_track=next((t for t in current if np.allclose(t._tlwh,
                      [det.box.x1,det.box.y1,det.box.width,det.box.height],atol=.01)),None)
        previous_id=self.target_id
        # Recover the original BoT state rather than replacing it with a new
        # tracklet's filter. The raw detection is applied exactly once after
        # verification. A duplicate tracklet created during the gap is retired.
        from .vendor.botsort.bot_sort import STrack
        selected=self.target
        candidate_id=None if candidate_track is None else candidate_track.track_id
        selected.re_activate(STrack(np.array([det.box.x1,det.box.y1,det.box.width,det.box.height]),det.confidence),
                             self.engine.frame_id,new_id=False)
        if candidate_track is not None and candidate_track is not selected:
            candidate_track.mark_removed()
            self.engine.tracked_stracks=[t for t in self.engine.tracked_stracks if t is not candidate_track]
            self.engine.lost_stracks=[t for t in self.engine.lost_stracks if t is not candidate_track]
        self.engine.lost_stracks=[t for t in self.engine.lost_stracks if t.track_id!=selected.track_id]
        self.engine.removed_stracks=[t for t in self.engine.removed_stracks if t.track_id!=selected.track_id]
        if all(t.track_id!=selected.track_id for t in self.engine.tracked_stracks):self.engine.tracked_stracks.append(selected)
        self.identity_event=dict(previous_track_id=previous_id,track_id=selected.track_id,
                                 candidate_track_id=candidate_id,
                                 logical_target_id=self.logical_target_id,reason='verified_reassociation',
                                 observations=p['hits'],strong_observations=p['strong'],relocation=relocation)
        self.association_reason='verified_returning_target' if relocation else 'verified_strong_recovery' if strong else 'verified_weak_recovery'
        self.pending=None;self.candidate_pending=False
        return selected

    def update(self,detections,source_time,now,frame_id,start_confidence=.45,
               strong_confidence=.25,continuation_confidence=.10,image=None):
        self.accepted=None;self.identity_event=None;self.candidate_pending=False;self.candidate_details=[]
        if not all(math.isfinite(t) for t in (source_time,now)) or source_time>now:
            self.association_reason='invalid_timestamp';return False
        if self.last_result_id is not None and (frame_id<=self.last_result_id or source_time<=self.last_result_time):
            self.association_reason='duplicate_or_out_of_order';return False
        if image is not None and image.shape[:2]!=(self.frame_size[1],self.frame_size[0]):
            raise ValueError('BoT-SORT image and detection coordinates must refer to the same source frame')
        if self.engine is None:
            from .vendor.botsort.bot_sort import BoTSORT
            args=SimpleNamespace(track_high_thresh=strong_confidence,track_low_thresh=low_threshold(strong_confidence),
                new_track_thresh=start_confidence,track_buffer=90,proximity_thresh=.5,appearance_thresh=.25,
                with_reid=False,cmc_method='none',name='follow-neo',ablation=False,mot20=False,match_thresh=.8)
            self.engine=BoTSORT(args);self.engine.gmc=CameraMotion()
        engine=self.engine
        self.acquisition_confidence=start_confidence
        engine.args.track_high_thresh=engine.track_high_thresh=strong_confidence
        engine.track_low_thresh=low_threshold(strong_confidence);engine.new_track_thresh=start_confidence
        dt=1. if self.last_result_time is None else max(.001,(source_time-self.last_result_time)*30.)
        engine.kalman_filter._motion_mat[:4,4:]=np.eye(4)*dt
        valid=[d for d in detections if math.isfinite(d.confidence) and engine.track_low_thresh<=d.confidence<=1
               and all(math.isfinite(v) for v in vars(d.box).values()) and d.box.width>0 and d.box.height>0]
        self.detector_count=len(valid)
        rows=np.array([[d.box.x1,d.box.y1,d.box.x2,d.box.y2,d.confidence] for d in valid],dtype=np.float64).reshape(-1,5)
        target_in_pool=self.target is not None and any(t is self.target for t in engine.tracked_stracks+engine.lost_stracks)
        tracks=engine.update(rows,image)
        if self.target is not None and not target_in_pool:
            # Retention by the application is separate from BoT's finite pool.
            # Advance this same BoT state once, never a second filter.
            from .vendor.botsort.bot_sort import STrack
            self.target.predict();STrack.multi_gmc([self.target],engine.gmc.last_warp)
        self.last_result_id,self.last_result_time=frame_id,source_time
        current=[t for t in tracks if t.frame_id==engine.frame_id]
        if self.target_id is None:
            candidates=[t for t in current if t.score>=start_confidence]
            if candidates:
                self.target=max(candidates,key=lambda t:t.score)
                self.target_id=self.logical_target_id=self.target.track_id
        selected=next((t for t in current if t.track_id==self.target_id),None)
        self.association_reason='matched_target' if selected is not None else 'no_target_match'
        if selected is None and self.confirmed:selected=self._recover(valid,current,source_time,strong_confidence)
        if selected is None or (not self.confirmed and selected.score<strong_confidence):
            if not self.confirmed:self.hits=0
            return False
        if self.association_reason=='matched_target':self.pending=None
        self.target=selected;self.target_id=selected.track_id
        self.pause_started=self.last_candidate_time=None
        x,y,w,h=map(float,selected._tlwh)
        self.accepted=Detection(Box(x,y,x+w,y+h),float(selected.score))
        self.last_detection=self.accepted
        self.measurement_time=source_time;self.measurement_id=frame_id
        self.measurement_weak=selected.score<strong_confidence
        if not self.measurement_weak:self.last_strong_time=source_time
        self.hits=min(2,self.hits+1);self.confirmed=self.confirmed or self.hits>=2
        self.history.append((source_time,np.array([x+w/2,y+h/2,w,h]),not self.measurement_weak))
        while self.history and source_time-self.history[0][0]>.30:self.history.popleft()
        return True

    def anchor(self):
        if self.target is None:return None
        m=self.target.mean
        return dict(model='botsort_cv',time=self.last_result_time,cx=float(m[0]),cy=float(m[1]),
                    width=float(max(2,m[2])),height=float(max(2,m[3])),vx=float(m[4]*30),vy=float(m[5]*30))

    def snapshot(self,now):
        a=self.anchor()
        if a is None:return Kinematics()
        x,y,vx,vy=project_center(a,now)
        return Kinematics(True,x,y,a['width'],a['height'],vx,vy)

    def uncertainty(self,now):
        if self.target is None:return dict(position_std_px=None,model='botsort')
        dt=max(0.,now-self.last_result_time)*30
        a=np.eye(8);a[:4,4:]=np.eye(4)*dt
        cov=a@self.target.covariance@a.T
        size=np.maximum(self.target.mean[2:4],1.)
        cov[:2,:2]+=np.diag((size/20)**2)*max(dt,dt*dt)
        return dict(position_std_px=np.sqrt(np.maximum(0,np.diag(cov)[:2])).tolist(),
                    age_seconds=max(0.,now-self.measurement_time) if self.measurement_time is not None else None,
                    model='botsort',velocity_units='pixels_per_second')

    def quality(self):
        result=dict(stable=False,motion_consistent=False,reason='insufficient_history',samples=len(self.history),span_ms=0.,residual_ratio=None)
        if len(self.history)<4:return result
        times=np.array([h[0] for h in self.history]);times-=times[-1]
        span=-times[0];result['span_ms']=float(span*1000)
        if span<.10:return result
        values=np.array([h[1] for h in self.history])
        slope,intercept=np.linalg.lstsq(np.c_[times,np.ones(len(times))],values[:,:2],rcond=None)[0]
        residual=np.abs(values[:,:2]-(times[:,None]*slope+intercept))
        ratio=float(np.max(np.percentile(residual,90,axis=0)/np.maximum(4*self.scale[:2],.12*np.median(values[:,2:],axis=0))))
        result['residual_ratio']=ratio
        velocity=np.array([self.snapshot(self.last_result_time).vx,self.snapshot(self.last_result_time).vy])
        if sum(h[2] for h in self.history)<3 or self.last_strong_time is None or self.measurement_time-self.last_strong_time>.10:result['reason']='weak_measurements'
        elif ratio>1 or np.any(np.max(values[:,2:],axis=0)/np.min(values[:,2:],axis=0)>1.6):result['reason']='inconsistent_boxes'
        elif np.any(abs(velocity)/np.array(self.frame_size)>.35):result['reason']='fast_image_motion'
        elif np.any(abs(velocity-slope)>np.maximum(30*self.scale[:2],.10*values[-1,2:]/span)):result['reason']='changing_direction'
        else:
            travel=(values[-1,:2]-values[0,:2])/self.scale[:2];steps=np.diff(values[:,:2],axis=0)/self.scale[:2]
            result.update(stable=True,reason='consistent_measurements',motion_consistent=bool(np.linalg.norm(travel)>8 and np.mean(steps@travel>0)>=.75))
        return result

    def diagnostics(self):
        return dict(backend=self.backend,target_id=self.target_id,logical_target_id=self.logical_target_id,reid=False,
                    motion_estimator='botsort_only',association_reason=self.association_reason,
                    candidate_pending=self.candidate_pending,candidate_details=self.candidate_details,
                    identity_changes=self.identity_changes,identity_event=self.identity_event,
                    detector_count=self.detector_count,target_state=None if self.target is None else int(self.target.state),
                    camera_warp=None if self.engine is None else self.engine.gmc.last_warp.tolist(),
                    camera_motion='initializing' if self.engine is None else self.engine.gmc.status,
                    camera_motion_inliers=0 if self.engine is None else self.engine.gmc.inliers)
