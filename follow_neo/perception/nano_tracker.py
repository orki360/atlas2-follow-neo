"""One CPU Nano worker, bounded history and latest-frame scheduling.

All image work occurs outside the ingress lock. No synthesized Detection enters
BoT-SORT. Guarded authority is applied separately from visual tracking.
"""
from collections import deque
from pathlib import Path
import hashlib
import threading
import time
import cv2
import numpy as np
from follow_neo.control.nano_guard import BRIDGE_SECONDS, VISUAL_SECONDS, overlap, appearance_unsupported, visual_quality_reason

WEIGHTS = {
    'nanotrack_backbone_sim.onnx': '530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4',
    'nanotrack_head_sim.onnx': '0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9',
}
# Visual continuity has a separate budget from authority to move the aircraft.
# Crossing a control deadline must not destroy the CPU track/template.
DISPLAY_SECONDS = .200
FRAME_GAP_SECONDS = .500


def create_tracker(root):
    folder = Path(root)/'models'/'nanotrack_v2'
    for name, digest in WEIGHTS.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError('NanoTrack weight checksum mismatch: '+name)
    params = cv2.TrackerNano_Params()
    params.backbone = str(folder/'nanotrack_backbone_sim.onnx')
    params.neckhead = str(folder/'nanotrack_head_sim.onnx')
    params.backend = cv2.dnn.DNN_BACKEND_OPENCV
    params.target = cv2.dnn.DNN_TARGET_CPU
    tracker=cv2.TrackerNano_create(params)
    # Pay first DNN kernel allocation before any real anchor. These pixels
    # never publish evidence and the next init replaces the dummy template.
    dummy=np.zeros((256,256,3),np.uint8)
    tracker.init(dummy,(96,96,32,32));tracker.update(dummy)
    return tracker


def xywh(box):
    x1, y1, x2, y2 = map(round, box)
    return x1, y1, max(2, x2-x1), max(2, y2-y1)


def historical_box(history, frame):
    """Interpolate only between already processed, nearby source frames.

    Latest-only scheduling skips frames; requiring an exact ID needlessly
    reinitialized DNN/replayed frames at almost every YOLO result.
    """
    if frame.frame_id in history:return history[frame.frame_id][1]
    before=[v for v in history.values() if 0<=frame.decoded_at-v[0]<=.05]
    after=[v for v in history.values() if 0<=v[0]-frame.decoded_at<=.05]
    if not before or not after:return None
    a=max(before,key=lambda v:v[0]);b=min(after,key=lambda v:v[0])
    if b[0]<=a[0]:return a[1]
    alpha=(frame.decoded_at-a[0])/(b[0]-a[0])
    return (np.asarray(a[1])*(1-alpha)+np.asarray(b[1])*alpha).tolist()


class TemplateRefresh:
    """Refresh Nano's fixed DNN template only from confirmed YOLO evidence."""
    def __init__(self):
        self.time=None;self.patch=None;self.size=None

    def applied(self, frame, detection):
        self.time=frame.decoded_at
        self.patch=AppearanceCheck.patch(frame.image,detection['box'])
        self.size=np.asarray(detection['box'][2:])-detection['box'][:2]

    def reason(self, frame, detection, agreement, armed):
        if (not armed or self.time is None or frame.decoded_at-self.time<.75
                or agreement<.65 or detection.get('confidence',0)<.80):return None
        size=np.asarray(detection['box'][2:])-detection['box'][:2]
        current=AppearanceCheck.patch(frame.image,detection['box'])
        if (min(size)<10 or current is None or self.patch is None or current.std()<8
                or cv2.Laplacian(current,cv2.CV_32F).var()<50):return None
        ratio=size/np.maximum(2,self.size)
        if np.any(ratio>1.4) or np.any(ratio<1/1.4):return 'confirmed_scale_change'
        a=current-current.mean();b=self.patch-self.patch.mean()
        den=float(np.linalg.norm(a)*np.linalg.norm(b))
        if den>100 and float(np.sum(a*b)/den)<.3:return 'confirmed_appearance_change'
        return None


class AppearanceCheck:
    """Independent, inexpensive cues; ambiguity never becomes new NEO proof."""
    def __init__(self):
        self.reference = self.mask = self.previous = self.box = None
        self.soft_failures = self.control_failures = self.texture_failures = 0
        self.unmatched_since = None

    @staticmethod
    def patch(image, box):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(round, box)
        crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            return None
        return cv2.cvtColor(cv2.resize(crop, (48,32)), cv2.COLOR_BGR2GRAY).astype(np.float32)

    def anchor(self, frame, detection):
        box = detection['box']; self.reference = self.patch(frame.image, box)
        x, y, w, h = xywh(box)
        mask = np.zeros((h,w), np.uint8)
        polys = [np.round(np.asarray(p)-[x,y]).astype(np.int32)
                 for p in detection.get('segments',[]) if len(p) >= 3]
        if polys:
            cv2.fillPoly(mask, polys, 255)
        self.mask = cv2.resize(mask, (48,32), interpolation=cv2.INTER_NEAREST) > 0
        if self.mask.sum() < 20:
            self.mask[:] = True
        self.soft_failures = self.control_failures = self.texture_failures = 0
        self.unmatched_since = None

    def flow(self, frame, box):
        if self.previous is None or self.box is None:
            return {}
        # Local shared coordinates, with background outside the previous BBOX.
        h, w = frame.image.shape[:2]
        if self.previous.image.shape != frame.image.shape:
            return {}
        a = self.box; b = box
        x = max(0, int(min(a[0],b[0])-48)); y = max(0,int(min(a[1],b[1])-48))
        xx = min(w,int(max(a[2],b[2])+48)); yy = min(h,int(max(a[3],b[3])+48))
        if xx<=x or yy<=y or max(xx-x,yy-y)>640:
            return {}
        first=cv2.cvtColor(self.previous.image[y:yy,x:xx],cv2.COLOR_BGR2GRAY)
        second=cv2.cvtColor(frame.image[y:yy,x:xx],cv2.COLOR_BGR2GRAY)
        points=cv2.goodFeaturesToTrack(first,80,.015,4)
        if points is None or len(points)<8:
            return {}
        following,status,_=cv2.calcOpticalFlowPyrLK(first,second,points,None,winSize=(15,15),maxLevel=2)
        back,valid,_=cv2.calcOpticalFlowPyrLK(second,first,following,None,winSize=(15,15),maxLevel=2)
        good=(status[:,0]>0)&(valid[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.5)
        p=points[:,0];delta=following[:,0]-p
        inside=(p[:,0]>=a[0]-x)&(p[:,0]<=a[2]-x)&(p[:,1]>=a[1]-y)&(p[:,1]<=a[3]-y)
        foreground=good&inside; background=good&~inside
        if self.mask is not None:
            mx=np.clip(((p[:,0]-(a[0]-x))*48/max(2,a[2]-a[0])).astype(int),0,47)
            my=np.clip(((p[:,1]-(a[1]-y))*32/max(2,a[3]-a[1])).astype(int),0,31)
            foreground &= self.mask[my,mx]
        if foreground.sum()<4 or background.sum()<8:
            return dict(flow_points=int(foreground.sum()),background_points=int(background.sum()))
        bg=np.median(delta[background],axis=0);fg=np.median(delta[foreground],axis=0)
        spread=float(np.median(np.linalg.norm(delta[foreground]-fg,axis=1)))
        displacement=(np.asarray(b[:2])+b[2:]-np.asarray(a[:2])-a[2:])*.5
        return dict(flow_points=int(foreground.sum()),background_points=int(background.sum()),
                    relative_motion_px=float(np.linalg.norm(fg-bg)),
                    foreground_flow_spread_px=spread,
                    box_flow_error_px=float(np.linalg.norm(displacement-fg)))

    def update(self, frame, box):
        metrics={};reason=None
        h,w=frame.image.shape[:2]
        if not all(np.isfinite(box)) or box[2]<=box[0] or box[3]<=box[1]:
            return 'invalid_box',metrics
        area=(box[2]-box[0])*(box[3]-box[1])
        visible=max(0,min(w,box[2])-max(0,box[0]))*max(0,min(h,box[3])-max(0,box[1]))/area
        metrics['visible_fraction']=visible
        if visible<.85:
            reason='target_leaving_frame'
        geometry=False
        if self.box is not None:
            old=np.array(self.box,float);new=np.array(box,float)
            ratio=(new[2:]-new[:2])/np.maximum(2,old[2:]-old[:2])
            dt=max(.001,frame.decoded_at-self.previous.decoded_at)
            jump=np.linalg.norm((new[:2]+new[2:]-old[:2]-old[2:])*.5)
            geometry=bool(np.any(ratio<.45) or np.any(ratio>2.2)
                          or jump>max(32,np.linalg.norm(old[2:]-old[:2])*1.5)*max(1,dt/.04))
            metrics.update(center_step_px=float(jump),geometry_suspect=geometry)
        patch=self.patch(frame.image,box)
        corr=None
        if patch is not None and self.reference is not None:
            a=self.reference[self.mask];b=patch[self.mask]
            ref_std=float(a.std());patch_std=float(b.std())
            metrics.update(reference_std=ref_std,patch_std=patch_std)
            a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b))
            if den>100:corr=float(np.dot(a,b)/den)
            lost_texture=ref_std>8 and patch_std<max(3.,.2*ref_std) and (corr is None or corr<.10)
            self.texture_failures=self.texture_failures+1 if lost_texture else 0
            metrics['texture_loss_streak']=self.texture_failures
            if self.texture_failures>=3:reason=reason or 'target_texture_lost'
            # A textured distractor (writing/window handle) can retain a high
            # Nano response. Persistent failure to match the last trusted
            # appearance must eventually end the visual track as well. This
            # is a visual-quality deadline, not the 250 ms YOLO control clock.
            if ref_std>8 and (corr is None or corr<.10):
                if self.unmatched_since is None:self.unmatched_since=frame.decoded_at
            elif ref_std<=8 or corr is not None and corr>=.30:
                self.unmatched_since=None
            unmatched=0. if self.unmatched_since is None else max(0.,frame.decoded_at-self.unmatched_since)
            metrics['appearance_unmatched_ms']=unmatched*1000
            if unmatched>=.6:reason=reason or 'appearance_unverified'
        metrics['appearance_correlation']=corr
        metrics.update(self.flow(frame,box))
        changed=corr is not None and corr<.10
        flow_error=metrics.get('box_flow_error_px',0)
        flow_conflict=flow_error>8
        # Background-like motion alone also describes a hovering drone.
        background=metrics.get('relative_motion_px',99)<1.3
        metrics['background_like_motion']=background
        # Do not count appearance change twice (once by itself and once inside
        # a background cue). Hovering/blur can satisfy both without drift.
        votes=int(geometry)+int(changed)+int(flow_conflict)
        self.control_failures=self.control_failures+1 if votes>=2 else 0
        metrics['control_suspect']=(self.control_failures>=2 or self.texture_failures>0
                                    or metrics.get('appearance_unmatched_ms',0)>=150)
        metrics['appearance_unsupported']=appearance_unsupported(metrics)
        metrics['control_suspect'] |= metrics['appearance_unsupported']
        # A small moving drone exposes textured background through its box and
        # blur. LK points there need not belong to the aircraft. A fixed 8 px
        # disagreement + appearance change blocked valid far targets. Require
        # coherent foreground flow and a size-relative discrepancy for LOSS;
        # the stricter control warning above is retained independently.
        diagonal=float(np.linalg.norm(np.asarray(box[2:])-box[:2]))
        reliable_flow=(metrics.get('flow_points',0)>=6
                       and metrics.get('foreground_flow_spread_px',99)<2.5
                       and metrics.get('relative_motion_px',0)>2.)
        loss_flow=reliable_flow and flow_error>max(12.,.5*diagonal)
        loss_votes=int(geometry)+int(changed)+int(loss_flow)
        self.soft_failures=self.soft_failures+1 if loss_votes>=2 else 0
        metrics['suspect_streak']=self.soft_failures
        if self.soft_failures>=2:reason=reason or 'appearance_or_background_drift'
        self.previous=frame;self.box=list(box)
        return reason,metrics


class NanoWorker:
    def __init__(self, root, mode='shadow', generation=0, wake=None, emit=None, factory=None, clock=time.monotonic):
        self.root=root;self.mode=mode;self.generation=generation
        self.wake=wake or (lambda:None);self.emit=emit or (lambda *args:None)
        self.factory=factory or (lambda:create_tracker(root));self.clock=clock
        self.lock=threading.Lock();self.event=threading.Event();self.stop_event=threading.Event()
        self.frames=deque(maxlen=20);self.pending=None;self.observation=None
        self.visual_history=deque(maxlen=60);self.update_times=deque(maxlen=60)
        self.epoch=0;self.identity=None;self.semantic_time=None;self.floor=clock()
        self.exit_context=None
        self.blocked=True;self.reason='waiting_strong_neo';self.hits=[];self.last_anchor_id=None
        self.updates=0;self.skipped=0;self.last_log=0.
        self.thread=threading.Thread(target=self._run,name='nano-visual',daemon=True)

    def start(self):self.thread.start()

    def reset(self, mode, generation):
        with self.lock:
            self.mode=mode;self.generation=generation;self.epoch+=1;self.floor=self.clock()
            self.pending=None;self.observation=None;self.identity=None;self.semantic_time=None
            self.exit_context=None
            self.visual_history.clear();self.update_times.clear()
            self.blocked=True;self.reason='waiting_strong_neo';self.hits=[];self.last_anchor_id=None
        self.event.set();self.wake()

    def submit_frame(self, frame):
        with self.lock:
            if self.mode=='off':return
            self.frames.append(frame)
            while self.frames and frame.decoded_at-self.frames[0].decoded_at>.35:self.frames.popleft()
        self.event.set()

    def submit_anchor(self, anchor):
        with self.lock:
            f=anchor['frame']
            if self.mode=='off' or anchor['generation']!=self.generation or f.decoded_at<=self.floor:return
            if self.last_anchor_id is not None and f.frame_id<=self.last_anchor_id:return
            self.exit_context=None
            if anchor['identity']!=self.identity:
                self.epoch+=1;self.identity=anchor['identity'];self.blocked=True
                self.reason='verifying_identity';self.hits=[];self.observation=None
                self.visual_history.clear();self.semantic_time=None
            self.last_anchor_id=f.frame_id
            self.hits=[t for t in self.hits if 0<f.decoded_at-t<=BRIDGE_SECONDS]+[f.decoded_at]
            self.pending=dict(anchor,epoch=self.epoch,arm=len(self.hits)>=2)
            info=dict(frame_id=f.frame_id,source_time=f.decoded_at,epoch=self.epoch,
                      identity=self.identity,strong_hits=len(self.hits),arm=self.pending['arm'])
        self.emit('nano_anchor',info)
        self.event.set()

    def veto(self, reason, now=None, expected_epoch=None, exit_context=None):
        now=self.clock() if now is None else now
        with self.lock:
            if expected_epoch is not None and expected_epoch!=self.epoch:return
            # Do not keep advancing the rearm floor on every aging tick.
            if self.blocked and self.reason==reason and self.semantic_time is None:return
            self.exit_context=(dict(exit_context,prior_epoch=self.epoch,identity=self.identity,
                                    generation=self.generation)
                               if reason=='target_leaving_frame' and exit_context else None)
            self.epoch+=1;self.floor=now;self.blocked=True;self.reason=reason
            self.pending=None;self.hits=[];self.observation=None;self.semantic_time=None
            self.visual_history.clear()
            epoch=self.epoch
        self.emit('nano_veto',dict(reason=reason,epoch=epoch,mode=self.mode,time=now));self.wake()

    def snapshot(self, now=None, *, source_time=None, frame_id=None):
        now=self.clock() if now is None else now
        with self.lock:
            result=dict(self.observation or {})
            result.update(mode=self.mode,epoch=self.epoch,generation=self.generation,identity=self.identity,
                          blocked=self.blocked,reason=self.reason,semantic_time=self.semantic_time,
                          updates=self.updates,skipped=self.skipped)
            result['exit_context']=dict(self.exit_context) if self.exit_context else None
            stamps=list(self.update_times)
            history=list(self.visual_history) if source_time is not None else []
        # Only actual Nano updates enter this history; a YOLO initialization
        # box is never independent evidence. Bracket skipped frames narrowly.
        result['source_alignment']=None
        if source_time is not None and frame_id is not None:
            before=[v for v in history if v['frame_id']<=frame_id and 0<=source_time-v['visual_time']<=.05]
            after=[v for v in history if v['frame_id']>=frame_id and 0<=v['visual_time']-source_time<=.05]
            if before and after:
                a=before[-1];b=after[0]
                good=all(not v.get('checks',{}).get('geometry_suspect',True)
                    and v.get('checks',{}).get('texture_loss_streak',0)==0
                    and v.get('checks',{}).get('visible_fraction',0)>=.98 for v in (a,b))
                if good:
                    span=b['visual_time']-a['visual_time']
                    alpha=(source_time-a['visual_time'])/span if span else 0.
                    box=[x+(y-x)*alpha for x,y in zip(a['box'],b['box'])]
                    result['source_alignment']=dict(box=box,source_time=source_time,frame_id=frame_id,
                        identity=result['identity'],generation=result['generation'],epoch=result['epoch'],
                        before_frame_id=a['frame_id'],after_frame_id=b['frame_id'],
                        before_time=a['visual_time'],after_time=b['visual_time'])
        if result['mode']=='off':return result
        semantic=result['semantic_time'];visual=result.get('visual_time')
        expired=('semantic_expired' if semantic is not None and not 0<=now-semantic<=BRIDGE_SECONDS
                 else 'visual_expired' if not result['blocked'] and (visual is None or not 0<=now-visual<=VISUAL_SECONDS) else None)
        if expired:result.update(blocked=True,reason=expired)
        quality_reason=visual_quality_reason(result.get('checks') or {})
        if quality_reason:
            result.update(blocked=True,reason=quality_reason)
        result['visual_active']=bool(result.get('box') and visual is not None and 0<=now-visual<=DISPLAY_SECONDS)
        result['visual_state']=('unverified' if result['blocked'] else 'verified') if result['visual_active'] else 'unavailable'
        result['update_fps']=((len(stamps)-1)/(stamps[-1]-stamps[0])
                              if len(stamps)>1 and stamps[-1]>stamps[0] and now-stamps[-1]<.5 else 0.)
        result['deadline']=min(semantic+BRIDGE_SECONDS,visual+VISUAL_SECONDS) if semantic is not None and visual is not None else None
        result['semantic_age_ms']=None if semantic is None else max(0,now-semantic)*1000
        result['visual_age_ms']=None if visual is None else max(0,now-visual)*1000
        return result

    def at_frame(self, frame):
        """Historical display only; never attach a future box to an old image."""
        with self.lock:
            entries=[v for v in self.visual_history if v['frame_id']<=frame.frame_id
                     and 0<=frame.decoded_at-v['visual_time']<=DISPLAY_SECONDS]
            if not entries:return None
            result=dict(entries[-1])
            result.update(mode=self.mode,identity=self.identity,epoch=self.epoch,
                          generation=self.generation,historical=True,display_only=True,
                          visual_active=True,visual_state='historical',blocked=True,
                          reason='historical_display_only')
            return result

    def evidence_at(self, frame):
        """Strict source-aligned evidence; display interpolation grants no authority."""
        with self.lock:
            if (self.mode!='guarded' or self.blocked or self.semantic_time is None
                    or frame.decoded_at<=self.floor
                    or not 0<frame.decoded_at-self.semantic_time<=.40):return None
            before=[v for v in self.visual_history if 0<=frame.decoded_at-v['visual_time']<=.05]
            after=[v for v in self.visual_history if 0<=v['visual_time']-frame.decoded_at<=.05]
            if not before or not after:return None
            a=max(before,key=lambda v:v['visual_time']);b=min(after,key=lambda v:v['visual_time'])
            for v in (a,b):
                checks=v.get('checks',{})
                score=v.get('score');corr=checks.get('appearance_correlation')
                if (not isinstance(score,(float,int)) or not np.isfinite(score) or score<.65
                        or not isinstance(corr,(float,int)) or not np.isfinite(corr) or corr<.30
                        or checks.get('control_suspect',True)
                        or checks.get('geometry_suspect',True)
                        or checks.get('visible_fraction',0)<.98):return None
            history={v['frame_id']:(v['visual_time'],v['box']) for v in (a,b)}
            box=historical_box(history,frame)
            if box is None:return None
            return dict(box=box,source_time=frame.decoded_at,frame_id=frame.frame_id,
                        identity=self.identity,generation=self.generation,epoch=self.epoch)

    def _run(self):
        tracker=None;epoch=-1;last_id=-1;last_time=None;history={};validator=AppearanceCheck();armed=False;low_scores=0
        if self.mode!='off':
            try:
                tracker=self.factory()
                self.emit('nano_ready',dict(backend='OpenCV CPU',mode=self.mode))
            except Exception as exc:
                self.veto('nano_error');self.emit('nano_error',dict(error=str(exc),mode=self.mode))
        while not self.stop_event.is_set():
            self.event.wait(.05);self.event.clear()
            with self.lock:
                mode=self.mode;token=self.epoch;generation=self.generation
                pending=self.pending;self.pending=None;frames=list(self.frames)
            if mode=='off':continue
            if token!=epoch:
                epoch=token;last_id=-1;last_time=None;history={};validator=AppearanceCheck();armed=False;low_scores=0
                template=TemplateRefresh()
            try:
                start=self.clock();cycle_start=time.perf_counter();replay=0;anchor_used=None;reinitialized=False
                if pending is not None and pending['epoch']==token:
                    src=pending['frame'];det=pending['detection'];anchor_used=src.frame_id
                    if start-src.decoded_at>.35:continue
                    previous=historical_box(history,src)
                    # A contradicting YOLO box is not allowed to teleport Nano.
                    if previous is not None and overlap(previous,det['box'])<.15:
                        self.emit('nano_visual_rejected',dict(reason='yolo_visual_disagreement',
                            frame_id=src.frame_id,source_time=src.decoded_at,visual_box=previous,
                            yolo_box=det['box'],epoch=token))
                        self.veto('yolo_visual_disagreement',expected_epoch=token);continue
                    agreement=overlap(previous,det['box']) if previous is not None else 0.
                    refresh_reason=template.reason(src,det,agreement,pending['arm'])
                    reinit=(last_id<0 or previous is None or agreement<.5 or refresh_reason is not None)
                    validator.anchor(src,det)
                    if reinit:
                        reinitialized=True
                        with self.lock:
                            self.visual_history.clear()
                        if tracker is None:tracker=self.factory()
                        ti=time.perf_counter()
                        tracker.init(src.image,xywh(det['box']))
                        template.applied(src,det)
                        self.emit('nano_anchor_applied',dict(frame_id=src.frame_id,reinitialized=True,
                            init_ms=(time.perf_counter()-ti)*1000,source_time=src.decoded_at,epoch=token,
                            template_reason=refresh_reason or 'acquire_or_realign',anchor_iou=agreement))
                        last_id=src.frame_id;last_time=src.decoded_at;low_scores=0
                        validator.previous=src;validator.box=list(det['box'])
                        history={last_id:(last_time,list(det['box']))}
                    armed=pending['arm']
                    with self.lock:
                        if token!=self.epoch or generation!=self.generation:continue
                        # The semantic clock advances only after checking the
                        # YOLO box against the track at its original frame.
                        self.semantic_time=src.decoded_at
                        # A matching delayed anchor must not revoke a newer,
                        # already checked camera-frame observation while waiting
                        # for the next frame. Reinitialization still revokes it.
                        retained=bool(armed and not reinitialized and self.observation
                            and not self.blocked and not self.observation.get('checks',{}).get('control_suspect'))
                        self.blocked=not retained
                        self.reason='visual_supported' if retained else 'waiting_visual_update' if armed else 'verifying_identity'
                # Initialization can be expensive. Read the newest camera
                # frame AFTER it, rather than processing an obsolete snapshot.
                with self.lock:frames=list(self.frames)
                if last_id<0 or not frames:continue
                candidates=[f for f in frames if f.frame_id>last_id]
                work=candidates[-1:]
                # A delayed YOLO seed can be several camera frames behind.
                # One intermediate update avoids jumping directly over a fast
                # target. This is bounded catch-up only after initialization;
                # normal tracking remains latest-only, without a growing queue.
                if reinitialized and len(candidates)>1 and candidates[-1].decoded_at-last_time>.08:
                    midpoint=(last_time+candidates[-1].decoded_at)*.5
                    middle=min(candidates[:-1],key=lambda f:abs(f.decoded_at-midpoint))
                    work=[middle,candidates[-1]]
                self.skipped+=max(0,len(candidates)-len(work))
                for frame in work:
                    if self.clock()-frame.decoded_at>DISPLAY_SECONDS:continue
                    if last_time is not None and frame.decoded_at-last_time>FRAME_GAP_SECONDS:
                        self.veto('visual_frame_gap',expected_epoch=token);break
                    tu=time.perf_counter();ok,box=tracker.update(frame.image);update_ms=(time.perf_counter()-tu)*1000
                    score=float(tracker.getTrackingScore())
                    low_scores=low_scores+1 if not np.isfinite(score) or score<.35 else 0
                    box=[float(box[0]),float(box[1]),float(box[0]+box[2]),float(box[1]+box[3])]
                    reason,metrics=validator.update(frame,box) if ok else ('tracker_update_failed',{})
                    if low_scores>=3:reason=reason or 'low_tracking_response'
                    last_id=frame.frame_id;last_time=frame.decoded_at;history[last_id]=(last_time,box)
                    history={i:b for i,b in history.items() if i>=last_id-30}
                    replay+=1;finished=self.clock()
                    if reason:
                        self.emit('nano_visual_rejected',dict(reason=reason,frame_id=last_id,
                            source_time=last_time,box=box,score=score if np.isfinite(score) else None,
                            checks=metrics,epoch=token))
                        context=dict(time=last_time,box=box,frame_size=[frame.image.shape[1],frame.image.shape[0]],
                                     checks=dict(metrics)) if reason=='target_leaving_frame' else None
                        self.veto(reason,finished,token,exit_context=context);break
                    with self.lock:
                        if token!=self.epoch or generation!=self.generation:break
                        self.updates+=1
                        self.observation=dict(box=box,frame_id=last_id,visual_time=last_time,completed_at=finished,
                            score=score if np.isfinite(score) else None,update_ms=update_ms,cycle_ms=(time.perf_counter()-cycle_start)*1000,
                            replay_frames=replay,anchor_frame_id=anchor_used,checks=metrics)
                        self.visual_history.append(dict(self.observation));self.update_times.append(finished)
                        if armed:self.blocked=False;self.reason='visual_supported'
                    self.wake()
                    if finished-self.last_log>=.1:
                        self.emit('nano_observation',self.snapshot(finished));self.last_log=finished
                    if time.perf_counter()-cycle_start>.045:break
            except Exception as exc:
                self.veto('nano_error',expected_epoch=token)
                self.emit('nano_error',dict(error=str(exc),mode=mode));tracker=None

    def stop(self):
        self.stop_event.set();self.event.set()
        if self.thread.ident:self.thread.join(timeout=2)
