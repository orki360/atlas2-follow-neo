"""Offline detector + BoT-SORT smoke replay on image frames; never opens aircraft I/O."""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from follow_neo.controller import FollowController
from follow_neo.detector import OnnxDroneDetector,DetectorSettings
from follow_neo.botsort_tracker import low_threshold
from follow_neo.models import resolve_model
from follow_neo.types import Box,Detection,Settings


def replay(folder,fps,limit):
    settings=Settings()
    path,manifest=resolve_model(ROOT)
    detector=OnnxDroneDetector(path,compute_mode='CPU',diagnostics_dir=ROOT/'logs')
    detector.settings=DetectorSettings(confidence=low_threshold(settings.confidence))
    controller=FollowController(tracker_backend='botsort')
    rows=[]
    files=sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))[:limit]
    if not files:raise ValueError('No image frames found')
    for fid,path in enumerate(files,1):
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Unreadable image: '+str(path))
        h,w=image.shape[:2];t=10+fid/fps
        result=detector.detect(image)
        ds=[Detection(Box(*d['box']),d['confidence']) for d in result['detections']]
        started=time.perf_counter()
        controller.observe(ds,t,t,fid,(w,h),settings,image=image)
        decision=controller.tick(t,w,h,t,settings)
        rows.append(dict(frame=path.name,detections=len(ds),accepted=decision['accepted'],
            tracking=decision['tracking'],state=decision['state'],
            tracker_ms=(time.perf_counter()-started)*1000,inference_ms=result['inference_ms']))
    costs=np.array([r['tracker_ms'] for r in rows[1:] or rows])
    return dict(scope='offline smoke test; no ground truth or flight validation',
        model=manifest['name'],frames=len(rows),fps_assumption=fps,
        accepted_frames=sum(r['accepted'] for r in rows),
        tracker_median_ms=float(np.median(costs)),tracker_p95_ms=float(np.percentile(costs,95)),rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames',type=Path);parser.add_argument('--fps',type=float,default=30.)
    parser.add_argument('--limit',type=int,default=60);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.fps<=0 or args.limit<1:parser.error('fps and limit must be positive')
    report=replay(args.frames,args.fps,args.limit)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
