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
        self.measured_reference=None
        self.candidate_rejections=[]
        self.acquisition=None
        self.acquisition_attempted=False

    def _acquire(self,valid,current,source_time,threshold):
        """A tentative ID is not a permanent target lock.

        Two coherent strong measurements confirm acquisition. Weak measurements
        may bridge a 150ms gap, but cannot start or confirm a target. Evidence
        expires after 600ms. Only the engine's existing Kalman state is selected.
        """
        old=self.acquisition
        if old and (source_time-old['time']>.15 or source_time-old['strong_times'][0]>.6):old=None
        self.acquisition=old
        strong=[d for d in valid if d.confidence>=threshold]
        nearby=[]
        if old:
            ref=old['box'];w,h=np.maximum(ref[2:],2.)
            for d in valid:
                ratios=np.array([d.box.width/w,d.box.height/h])
                delta=np.array([d.box.cx,d.box.cy])-ref[:2]
                gate=np.maximum(1.,np.minimum(np.array(self.frame_size)*.22,np.maximum([w*2.,h*2.],16.)))
                if np.linalg.norm(delta/gate)<=1 and np.all(ratios>=.5) and np.all(ratios<=2.):nearby.append(d)
        candidates=nearby if nearby else strong
        if not nearby:old=None
        if len(candidates)>1:
            self.acquisition=None;self.hits=0;self.association_reason='acquisition_ambiguous'
            return None
        if not candidates:
            # An empty frame does not erase recent evidence, or retain a dead ID.
            self.hits=0 if self.acquisition is None else len(self.acquisition['strong_times'])
            self.association_reason='acquisition_gap' if self.acquisition else 'acquisition_retry' if self.acquisition_attempted else 'waiting_for_acquisition'
            return None
        det=candidates[0]
        if old is None and det.confidence<threshold:return None
        times=[] if old is None else list(old['strong_times'])
        if det.confidence>=threshold:times.append(source_time)
        self.acquisition=dict(box=np.array([det.box.cx,det.box.cy,det.box.width,det.box.height]),
                              time=source_time,strong_times=times)
        self.acquisition_attempted=True;self.hits=len(times)
        self.association_reason='acquisition_verifying'
        if det.confidence<threshold:return None
        selected=next((t for t in current if np.allclose(t._tlwh,
            [det.box.x1,det.box.y1,det.box.width,det.box.height],atol=.01)),None)
        if selected is None:return None
        self.target=selected;self.target_id=self.logical_target_id=selected.track_id
        self.confirmed=len(times)>=2
        if self.confirmed:
            self.acquisition=None;self.association_reason='acquisition_confirmed'
        return selected

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
        # A real measurement carried by camera motion is a second geometric
        # reference, NOT a second motion estimator. It cannot drift with velocity.
        ref=self.measured_reference
        if ref is not None and age<=RECOVERY_SECONDS:
            rw,rh=np.maximum(ref[2:],2.)
            rr=np.array([box.width/rw,box.height/rh])
            clipped=(ref[0]-rw/2<=self.frame_size[0]*.02 or ref[0]+rw/2>=self.frame_size[0]*.98
                     or ref[1]-rh/2<=self.frame_size[1]*.02 or ref[1]+rh/2>=self.frame_size[1]*.98)
            lower,upper=(.25,3.) if clipped else (.45,2.2)
            rg=np.minimum(np.array(self.frame_size)*.25,np.maximum([rw*2.5,rh*2.5],24.))
            rd=float(np.linalg.norm((np.array([box.cx,box.cy])-ref[:2])/rg))
            if rd<=1 and np.all(rr>=lower) and np.all(rr<=upper):
                return .15+rd+.15*float(np.sum(np.abs(np.log(rr))))
        # Maintain a coherent candidate through a brief detector gap or weak
        # confidence dip, even if the old predicted track has moved off-screen.
        old=self.pending
        if old and 0<now-old['time']<=.15:
            point=old['point']
            if (np.linalg.norm(np.array([box.cx,box.cy])-point)<=max(16.,box.width,box.height)*1.25
                and .55<=box.width/old['width']<=1.8 and .45<=box.height/old['height']<=2.2):
                if old['relocation']:
                    if old['strong']>=1:return 2.1
                elif age<=RECOVERY_SECONDS:return 1.1
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
        ranked_ids={id(d) for _,d in ranked}
        self.candidate_rejections=[dict(box=list(vars(d.box).values()),reason='outside_geometry_or_weak_return')
                                   for d in valid if id(d) not in ranked_ids]
        if not ranked:
            if self.pending and source_time-self.pending['time']<=.15:
                self.association_reason='candidate_gap'
                self.candidate_pending=bool(self.pause_started is not None and source_time-self.pause_started<=.8)
            else:self.pending=None;self.association_reason='no_plausible_candidate'
            return None
        if len(ranked)>1 and ranked[1][0]-ranked[0][0]<.20:
            self.pending=None;self.association_reason='ambiguous_candidates';return None
        cost,det=ranked[0];old=self.pending;relocation=cost>=2.
        if self.pause_started is None:
            self.pause_started=source_time
        self.last_candidate_time=source_time
        if old is not None:relocation=relocation or old['relocation']
        point=np.array([det.box.cx,det.box.cy]);coherent=False;step=None
        if old is not None and 0<source_time-old['time']<=.15:
            previous=old['point'];step=point-previous
            scale=max(16.,det.box.width,det.box.height)
            coherent=(np.linalg.norm(step)<=scale*1.5 and .55<=det.box.width/old['width']<=1.8
                      and .45<=det.box.height/old['height']<=2.2)
            if old['step'] is not None and np.linalg.norm(step)>scale*.15:
                coherent=coherent and float(step@old['step'])>=-scale*scale*.1
        if not coherent:old=None
        samples=[] if old is None else [s for s in old['samples'] if source_time-s[0]<=.6]
        samples.append((source_time,det.confidence>=strong_confidence))
        self.pending=dict(point=point,width=det.box.width,height=det.box.height,time=source_time,relocation=relocation,
                          since=samples[0][0],hits=len(samples),strong=sum(s[1] for s in samples),samples=samples,
                          step=step if coherent else None)
        p=self.pending;self.candidate_pending=source_time-self.pause_started<=.8
        self.association_reason='candidate_verifying'
        strong=p['strong']>=3
        needed=8 if relocation else 3 if strong else 4
        span=.25 if relocation else .06 if strong else .10
        if p['hits']<needed or source_time-p['since']<span or relocation and p['strong']<3:return None
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
        self.accepted=None;self.identity_event=None;self.candidate_pending=False;self.candidate_details=[];self.candidate_rejections=[]
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
        if self.acquisition is not None:
            warp=engine.gmc.last_warp;ref=self.acquisition['box']
            ref[:2]=warp[:,:2]@ref[:2]+warp[:,2]
            ref[2:]=np.abs(warp[:,:2])@ref[2:]
        if self.pending is not None:
            warp=engine.gmc.last_warp
            self.pending['point']=warp[:,:2]@self.pending['point']+warp[:,2]
            if self.pending['step'] is not None:self.pending['step']=warp[:,:2]@self.pending['step']
        if self.measured_reference is not None:
            warp=engine.gmc.last_warp
            self.measured_reference[:2]=warp[:,:2]@self.measured_reference[:2]+warp[:,2]
            self.measured_reference[2:]=np.abs(warp[:,:2])@self.measured_reference[2:]
        if self.target is not None and not target_in_pool:
            # Retention by the application is separate from BoT's finite pool.
            # Advance this same BoT state once, never a second filter.
            from .vendor.botsort.bot_sort import STrack
            self.target.predict();STrack.multi_gmc([self.target],engine.gmc.last_warp)
        self.last_result_id,self.last_result_time=frame_id,source_time
        current=[t for t in tracks if t.frame_id==engine.frame_id]
        acquiring=not self.confirmed
        if acquiring:
            selected=self._acquire(valid,current,source_time,max(start_confidence,strong_confidence))
            if selected is None:
                self.target=None;self.target_id=self.logical_target_id=None
                return False
        else:
            selected=next((t for t in current if t.track_id==self.target_id),None)
            self.association_reason='matched_target' if selected is not None else 'no_target_match'
            if selected is None:selected=self._recover(valid,current,source_time,strong_confidence)
            if selected is None:return False
        if self.association_reason=='matched_target':self.pending=None
        self.target=selected;self.target_id=selected.track_id
        self.pause_started=self.last_candidate_time=None
        x,y,w,h=map(float,selected._tlwh)
        self.accepted=Detection(Box(x,y,x+w,y+h),float(selected.score))
        self.last_detection=self.accepted
        self.measured_reference=np.array([x+w/2,y+h/2,w,h])
        self.measurement_time=source_time;self.measurement_id=frame_id
        self.measurement_weak=selected.score<strong_confidence
        if not self.measurement_weak:self.last_strong_time=source_time
        if not acquiring:self.hits=min(2,self.hits+1)
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
                    acquisition_state='confirmed' if self.confirmed else 'verifying' if self.acquisition else 'reacquiring' if self.acquisition_attempted else 'waiting',
                    acquisition_strong_hits=self.hits if not self.confirmed else 0,
                    motion_estimator='botsort_only',association_reason=self.association_reason,
                    candidate_pending=self.candidate_pending,candidate_details=self.candidate_details,
                    candidate_rejections=self.candidate_rejections,
                    candidate_hits=0 if self.pending is None else self.pending['hits'],
                    candidate_strong_hits=0 if self.pending is None else self.pending['strong'],
                    candidate_span_ms=0 if self.pending is None else (self.pending['time']-self.pending['since'])*1000,
                    candidate_pause_budget_seconds=.8,
                    identity_changes=self.identity_changes,identity_event=self.identity_event,
                    detector_count=self.detector_count,target_state=None if self.target is None else int(self.target.state),
                    camera_warp=None if self.engine is None else self.engine.gmc.last_warp.tolist(),
                    camera_motion='initializing' if self.engine is None else self.engine.gmc.status,
                    camera_motion_inliers=0 if self.engine is None else self.engine.gmc.inliers)
