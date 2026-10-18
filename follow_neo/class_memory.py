"""Bounded class evidence for one confirmed NEO, never a new target detector."""
from collections import deque
import math
from .types import Detection

EVIDENCE_GAP = .30
EVIDENCE_WINDOW = 1.20
CLASS_BRIDGE_SECONDS = .30
CLASS_CONTROL_SECONDS = .45


def overlap(a,b):
    area=max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
    union=a.width*a.height+b.width*b.height-area
    return area/union if union>0 else 0.


class NeoClassMemory:
    def __init__(self):
        self.identity=None;self.votes=deque();self.last_id=None;self.box=None
        self.last_strong=None;self.last_real_time=None;self.diagnostics={}

    def remember(self,tracker,detections,source_time,frame_id,threshold):
        if not tracker.confirmed or tracker.accepted is None:return
        identity=tracker.logical_target_id
        if identity!=self.identity:
            self.__init__();self.identity=identity
        if frame_id==self.last_id:return
        chosen=tracker.accepted.box
        neo=[d for d in detections if d.class_id==0 and d.class_name=='neo' and overlap(d.box,chosen)>.7]
        if not neo:return
        best=max(neo,key=lambda d:d.confidence)
        mini=max((d.confidence for d in detections if d.class_id==1 and overlap(d.box,best.box)>.5),default=0.)
        if self.last_strong is not None and source_time-self.last_strong>EVIDENCE_GAP:self.votes.clear()
        if self.box is not None and overlap(self.box,best.box)<.3:self.votes.clear()
        while self.votes and source_time-self.votes[0]>EVIDENCE_WINDOW:self.votes.popleft()
        self.last_id=frame_id;self.box=best.box;self.last_real_time=source_time
        # Only real, strong NEO wins build the lock. Repeated ticks, weak boxes
        # and history-resolved measurements cannot vote or renew the deadline.
        if best.confidence>=threshold and best.confidence>=mini:
            self.votes.append(source_time);self.last_strong=source_time

    def resolve(self,detections,tracker,source_time,threshold):
        self.diagnostics=dict(locked=False,neo_wins=len(self.votes),supported=False,
                              last_strong_neo_time=self.last_strong,reason='waiting_neo_evidence')
        valid=(tracker.confirmed and self.identity==tracker.logical_target_id
               and len(self.votes)>=3 and self.last_strong is not None
               and 0<=source_time-self.last_strong<=CLASS_BRIDGE_SECONDS)
        if not valid:return None,None
        self.diagnostics.update(locked=True,reason='recent_neo_identity')
        if any(d.class_id==0 and d.class_name=='neo' for d in detections):return None,None
        predicted=tracker.snapshot(source_time).box
        candidates=[]
        for d in detections:
            scores=d.class_scores
            if d.class_id!=1 or d.class_name!='MINI PRO4' or len(scores)!=2:continue
            neo,mini=scores
            if not all(math.isfinite(s) and 0<=s<=1 for s in scores):continue
            # Clear MINI evidence never becomes NEO. Only a near-tie containing
            # an explicit NEO score may support a short, geometry-bound bridge.
            if not (.10<=neo<mini and mini-neo<=.08):continue
            if self.box is None or overlap(d.box,predicted)<.55 or overlap(d.box,self.box)<.45:continue
            if not (.75<=d.box.width/max(2.,self.box.width)<=1.33 and
                    .75<=d.box.height/max(2.,self.box.height)<=1.33):continue
            candidates.append(d)
        if len(candidates)!=1:
            self.diagnostics['reason']='no_unique_ambiguous_box';return None,None
        raw=candidates[0]
        if sum(overlap(d.box,raw.box)>.15 for d in detections)>1:
            self.diagnostics['reason']='competing_overlapping_objects';return None,None
        support=dict(raw_class_id=raw.class_id,raw_class_name=raw.class_name,
                     raw_class_scores=list(raw.class_scores),resolved_class_id=0,resolved_class_name='neo',
                     source='temporal_identity',logical_target_id=self.identity,
                     last_real_neo_time=self.last_real_time,
                     last_strong_neo_time=self.last_strong,expires_at=self.last_strong+CLASS_CONTROL_SECONDS)
        self.diagnostics.update(supported=True,reason='bounded_class_ambiguity',**support)
        # Conservative weak association weight; no acquisition/strong evidence.
        resolved=Detection(raw.box,min(raw.class_scores[0],threshold*.99),0,'neo',raw.model_class_name,raw.class_scores)
        return resolved,support
