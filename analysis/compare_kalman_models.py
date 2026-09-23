"""Offline forecast comparison on real accepted YOLO centers (not ground truth).

Withhold the next measurements for 0.1/0.2/0.3 s in dense detection segments.
A constant-acceleration candidate is evaluated only here, never in flight.
"""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_neo.tracker import BBoxTracker
from follow_neo.types import Detection,Box


class AccelerationCandidate:
    def __init__(self,size,jerk=300.):
        self.k=cv2.KalmanFilter(6,2,0,cv2.CV_64F);self.size=size;self.jerk=jerk;self.time=None
        self.k.measurementMatrix=np.eye(2,6,dtype=np.float64)
    def update(self,box,confidence,stamp):
        z=np.array([box.cx,box.cy],np.float64).reshape(2,1)
        scale=np.array(self.size)/[640,480]
        if self.time is None:
            self.k.statePost=np.r_[z,np.zeros((4,1))]
            self.k.errorCovPost=np.diag(np.tile(scale,3)**2*np.array([12,12,1200,1200,30000,30000]))
        else:
            dt=stamp-self.time;a=np.eye(6);a[0,2]=a[1,3]=dt
            a[0,4]=a[1,5]=dt*dt/2;a[2,4]=a[3,5]=dt
            q=np.zeros((6,6))
            for axis in (0,1):
                inds=[axis,axis+2,axis+4]
                block=np.array([[dt**5/20,dt**4/8,dt**3/6],
                                [dt**4/8,dt**3/3,dt**2/2],
                                [dt**3/6,dt**2/2,dt]])*(self.jerk*scale[axis])**2
                q[np.ix_(inds,inds)]=block
            self.k.transitionMatrix=a;self.k.processNoiseCov=q;self.k.predict()
            std=np.maximum(3*scale,.04*np.array([box.width,box.height]))
            self.k.measurementNoiseCov=np.diag(std**2)/max(.25,min(1.,confidence))
            self.k.correct(z)
        self.time=stamp
    def predict(self,stamp):
        dt=stamp-self.time;x=self.k.statePost[:,0]
        return x[:2]+x[2:4]*dt+x[4:6]*dt*dt/2


def collect(events):
    sequences=[];seq=[];last_id=None;last_time=None;size=None
    for line in Path(events).open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']=='tracking_reset':
            if len(seq)>=15:sequences.append((size,seq))
            seq=[];last_time=None
        if e['event']!='observation':continue
        d=e['decision'];mid=d.get('measurement_id')
        if not d.get('accepted') or mid!=e.get('result_frame_id') or mid==last_id:continue
        size=tuple(d['frame_size']);stamp=d['measurement_time'];last_id=mid
        detections=e.get('detections',[])
        if not detections:continue
        # Associate to the recorded filtered center. YOLO remains the reference,
        # so these scores include detector jitter and are not metric 3-D errors.
        k=d['kinematics']
        obj=min(detections,key=lambda a:abs((a['box'][0]+a['box'][2])/2-k['cx'])+abs((a['box'][1]+a['box'][3])/2-k['cy']))
        if obj['confidence']<.4:continue
        if last_time is not None and (stamp-last_time>.08 or stamp<=last_time):
            if len(seq)>=15:sequences.append((size,seq))
            seq=[]
        seq.append((stamp,Box(*obj['box']),obj['confidence']));last_time=stamp
    if len(seq)>=15:sequences.append((size,seq))
    return sequences


def compare(events,baseline=None):
    base_tracker=None
    if baseline is not None:
        import importlib.util,importlib
        package=Path(baseline)/"follow_neo"
        spec=importlib.util.spec_from_file_location("update8_reference",package/"__init__.py",submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        base_tracker=importlib.import_module("update8_reference.tracker").BBoxTracker
    buckets={name:{str(h):[] for h in (.1,.2,.3)} for name in ('update9_cv','ca_jerk150','ca_jerk300','ca_jerk600')}
    if base_tracker:buckets['update8_cv']={str(h):[] for h in (.1,.2,.3)}
    sequences=collect(events)
    for size,seq in sequences:
        cv=BBoxTracker(size);old=base_tracker(size) if base_tracker else None
        ca=[AccelerationCandidate(size,j) for j in (150.,300.,600.)]
        for i,(stamp,box,confidence) in enumerate(seq):
            cv.update([Detection(box,confidence)],stamp,stamp,i+1)
            if old:old.update([Detection(box,confidence)],stamp,stamp,i+1)
            for model in ca:model.update(box,confidence,stamp)
            if i<10 or i%3:continue
            for horizon in (.1,.2,.3):
                future=next((r for r in seq[i+1:] if r[0]-stamp>=horizon),None)
                if future is None or future[0]-stamp>horizon+.05:continue
                when,b,_=future;truth=np.array([b.cx,b.cy]);k=cv.snapshot(when)
                points=[np.array([k.cx,k.cy])]+[m.predict(when) for m in ca]
                if old:
                    prior=old.snapshot(when);points.append(np.array([prior.cx,prior.cy]))
                for name,point in zip(buckets,points):
                    # Diagonal-relative error makes mixed resolutions comparable.
                    error=float(np.linalg.norm(point-truth)/math.hypot(*size)*100)
                    buckets[name][str(horizon)].append(error)
    return dict(reference='accepted YOLO centers; no independent ground truth',dense_segments=len(sequences),
        metric='center forecast error as percent of image diagonal',
        results={name:{h:dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95)))
                 for h,v in values.items() if v} for name,values in buckets.items()})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('events',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument("--baseline",type=Path,help="Optional Update 8 project directory for comparison")
    args=ap.parse_args();report=json.dumps(compare(args.events,args.baseline),indent=2)
    if args.output:args.output.write_text(report,encoding='utf-8')
    print(report)
