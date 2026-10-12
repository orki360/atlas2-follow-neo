"""Sequential, same-frame ROI classification before target acquisition."""
from dataclasses import dataclass, asdict, replace
import math
import time
import numpy as np


@dataclass(frozen=True)
class RoiSettings:
    enabled: bool = True
    width: int = 960
    height: int = 540
    max_box_width: int = 480
    max_box_height: int = 270
    confirmations: int = 2
    confidence: float = .50
    match_iou: float = .30
    max_gap_seconds: float = .75
    recheck_seconds: float = .25

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('ROI enabled must be a boolean')
        for name, lo, hi in [('width',32,1920),('height',32,1080),
                             ('max_box_width',1,1920),('max_box_height',1,1080),('confirmations',1,10)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f'ROI {name} must be an integer between {lo} and {hi}')
        if self.max_box_width > self.width or self.max_box_height > self.height:
            raise ValueError('ROI entry thresholds cannot exceed the crop dimensions')
        for name,lo,hi in [('confidence',.05,.99),('match_iou',.05,.95),('max_gap_seconds',.1,2.),('recheck_seconds',0.,1.)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'ROI {name} must be between {lo} and {hi}')
        return self

    @classmethod
    def from_config(cls, config):
        values = {k:v for k,v in config.get('roi',{}).items() if k in asdict(cls())}
        return cls(**values).validate()


def box_iou(a, b):
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0.,x[2]-x[0])*max(0.,x[3]-x[1])
    union=area(a)+area(b)-overlap
    return overlap/union if union>0 else 0.


def crop_bounds(box, image_shape, settings):
    h,w=image_shape[:2]
    if w<settings.width or h<settings.height:return None
    x=max(0,min(w-settings.width,int(round((box[0]+box[2]-settings.width)/2))))
    y=max(0,min(h-settings.height,int(round((box[1]+box[3]-settings.height)/2))))
    return [x,y,x+settings.width,y+settings.height]


def translated_detection(detection, x, y):
    item=dict(detection)
    item['box']=[float(v)+(x if i%2==0 else y) for i,v in enumerate(detection['box'])]
    if 'segments' in detection:
        item['segments']=[[[float(px)+x,float(py)+y] for px,py in polygon] for polygon in detection['segments']]
    return item


def roi_cache_allowed(decision, now):
    """Only fresh, accepted tracking may reuse a previous crop classification."""
    if not isinstance(decision,dict):return False
    if (decision.get('state')!='TRACK' or decision.get('confirmed') is not True
        or decision.get('accepted') is not True or decision.get('stale')):return False
    for key,limit in (('prediction_time',.20),('measurement_time',.25)):
        stamp=decision.get(key)
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not 0<=now-stamp<=limit:return False
    return True


class RoiVerifier:
    def __init__(self):
        self.key=None
        self.tracks={}
        self.next_id=1
        self.last_frame=None
        self.last_result=None

    def process(self, frame, result, detector, settings, *, generation=0, cancelled=lambda:False,
                allow_cached=False, cache_confidence=1.):
        settings.validate()
        key=(generation,settings,frame.image.shape[:2])
        if self.key!=key:
            self.key=key;self.tracks={};self.last_frame=None;self.last_result=None
        if self.last_frame==frame.frame_id:return self.last_result
        names=detector.class_names
        supported=isinstance(names,dict) and {'NEO','MINI PRO4'}.issubset(names.values())
        output=dict(result, full_detections=result['detections'],
                    detections=[dict(d) for d in result['detections']], rois=[],
                    roi_ms=0.,roi_inferences=0,roi_cached=0,
                    full_inference_ms=result.get('inference_ms',0.),roi_enabled=settings.enabled and supported,
                    roi_note='enabled' if settings.enabled and supported else 'disabled' if not settings.enabled else 'requires NEO / MINI PRO4 model')
        if not settings.enabled or not supported:
            self.tracks={}
            return output
        candidates=[]
        for index,d in enumerate(output['detections']):
            if d.get('model_class_name')!='NEO' or not d.get('tracking_eligible',True):continue
            box=d['box']
            if box[2]-box[0]>settings.max_box_width or box[3]-box[1]>settings.max_box_height:
                d['roi_status']='skipped_size';continue
            bounds=crop_bounds(box,frame.image.shape,settings)
            if bounds is None:
                # No rescaling of a smaller source into a fictitious full-size crop.
                d.update(roi_status='source_too_small',tracking_eligible=False);continue
            d.update(tracking_eligible=False,roi_status='pending')
            candidates.append((index,d,bounds))
        prior={k:t for k,t in self.tracks.items() if 0<frame.decoded_at-t['time']<=settings.max_gap_seconds}
        links={index:[k for k,t in prior.items() if box_iou(d['box'],t['box'])>=settings.match_iou]
               for index,d,_ in candidates}
        next_tracks={}
        started=time.perf_counter()
        original_settings=detector.settings
        try:
            detector.settings=replace(original_settings,confidence=settings.confidence)
            for index,d,bounds in candidates:
                if cancelled():
                    d['roi_status']='cancelled';continue
                possible=links[index]
                track=None
                if len(possible)==1 and sum(possible[0] in ids for ids in links.values())==1:
                    track=prior[possible[0]]
                if track is None:
                    track=dict(id=self.next_id,hits=0);self.next_id+=1
                verified_at=track.get('verified_at')
                cached=(allow_cached and track['hits']>=settings.confirmations
                        and d['confidence']>=cache_confidence and verified_at is not None
                        and 0<=frame.decoded_at-verified_at<settings.recheck_seconds)
                if cached:
                    d.update(tracking_eligible=True,roi_status='approved_cached',roi_id=track['id'],
                             roi_hits=track['hits'],roi_verified_at=verified_at,
                             confidence=min(d['confidence'],track['verified_confidence']))
                    track.update(box=list(d['box']),time=frame.decoded_at)
                    next_tracks[track['id']]=track;output['roi_cached']+=1
                    # Never draw an old crop measurement on a newer source frame.
                    output['rois'].append(dict(id=track['id'],box=bounds,candidate_box=list(d['box']),
                        status='approved_cached',hits=track['hits'],required=settings.confirmations,
                        verified_at=verified_at,detections=[]))
                    continue
                x1,y1,x2,y2=bounds
                local=detector.detect(frame.image[y1:y2,x1:x2])  # Same detector, serial calls only.
                output['roi_inferences']+=1
                for field in ('inference_ms','preprocess_ms','postprocess_ms'):
                    output[field]=output.get(field,0.)+local.get(field,0.)
                mapped=[translated_detection(v,x1,y1) for v in local['detections']]
                matches=[v for v in mapped if v['confidence']>=settings.confidence
                         and box_iou(d['box'],v['box'])>=settings.match_iou]
                matches.sort(key=lambda v:box_iou(d['box'],v['box']),reverse=True)
                # Competing boxes/classes do not constitute verification of this candidate.
                match=matches[0] if len(matches)==1 else None
                status='unresolved'
                if match and match.get('model_class_name')=='NEO':
                    track['hits']=min(settings.confirmations,track['hits']+1)
                    status='approved' if track['hits']>=settings.confirmations else 'pending'
                    d['tracking_eligible']=status=='approved'
                    d['confidence']=min(d['confidence'],match['confidence'])
                    track.update(verified_at=frame.decoded_at,verified_confidence=match['confidence'])
                else:
                    track['hits']=0
                    if match and match.get('model_class_name')=='MINI PRO4':
                        status='rejected_mini'
                        d.update(class_id=match['class_id'],model_class_name='MINI PRO4',confidence=match['confidence'])
                # Retain the full-frame box for the existing distance/control contract.
                d.update(roi_status=status,roi_id=track['id'],roi_hits=track['hits'])
                track.update(box=list(d['box']),time=frame.decoded_at)
                next_tracks[track['id']]=track
                output['rois'].append(dict(id=track['id'],box=bounds,candidate_box=list(d['box']),
                                          status=status,hits=track['hits'],required=settings.confirmations,
                                          detections=mapped))
        finally:
            detector.settings=original_settings
        self.tracks=next_tracks
        output['roi_ms']=(time.perf_counter()-started)*1000
        output['total_ms']=result.get('total_ms',0.)+output['roi_ms']
        self.last_frame=frame.frame_id;self.last_result=output
        return output
