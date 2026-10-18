"""Offline tracker replay using saved detections, timestamps and clean frames.

Camera/aircraft/transport behavior is fixed by the recording, not simulated.
No model inference, UI or network connections are started.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from follow_neo.controller import FollowController
from follow_neo.types import Box,Detection,Settings


def rows(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            if line.strip():yield json.loads(line)


def replay(run,output):
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError('Output must be outside the evidence run')
    folder=next(run.glob('recording_*'))
    recorded=list(rows(folder/'frames.jsonl'))
    indices={f['frame_id']:i for i,f in enumerate(recorded)}
    start,end=recorded[0]['decoded_at'],recorded[-1]['decoded_at']
    capture=cv2.VideoCapture(str(folder/'clean.mp4'))
    if not capture.isOpened():raise RuntimeError('Cannot open local clean video')
    controller=FollowController('botsort',0,'neo');settings=Settings()
    counts=Counter();old_counts=Counter();states=Counter();checks=[]
    last_id=None;index=-1;image=None
    cv2.setNumThreads(2)
    try:
        for row in rows(run/'events.jsonl'):
            event=row['event']
            if event in ('session_start','settings','search_settings'):
                settings=Settings(**row['settings']).validate()
            if not start<=row['monotonic']<=end+.75:continue
            if event=='control_restart':controller.restart_control()
            if event=='tracking_reset':controller.reset()
            if event!='observation':continue
            frame_id=row.get('result_frame_id')
            if frame_id==last_id or frame_id not in indices:continue
            last_id=frame_id;desired=indices[frame_id]
            while index<desired:
                ok,image=capture.read();index+=1
                if not ok:raise RuntimeError('Video ended before sidecar frame')
            ds=[Detection(Box(*d['box']),d['confidence'],d.get('class_id'),d.get('class_name'),
                          d.get('model_class_name'),tuple(d.get('class_scores',()))) for d in row['detections']]
            old=row['decision'];source=row['decoded_at'];now=old['prediction_time']
            # Use the recorded result-ready time for arrival, then original
            # decision time for aging, with actual saved camera pixels for GMC.
            arrival=max(source,min(now,row.get('result_ready_at',now)))
            controller.observe(ds,source,arrival,frame_id,(image.shape[1],image.shape[0]),settings,image)
            control=old.get('control_status',{})
            d=controller.tick(now,image.shape[1],image.shape[0],source,settings,old.get('heading'),
                              control_available=control.get('active',False) and control.get('mode')=='DANCE')
            strong=any(x.class_id==0 and x.confidence>=settings.confidence for x in ds)
            for value,target in ((old,old_counts),(d,counts)):
                target['results']+=1;target['accepted']+=bool(value.get('accepted'))
                target['strong_neo_rejected']+=strong and not value.get('accepted')
                target['class_memory_measurements']+=bool(value.get('accepted') and value.get('class_support'))
            states[d['state']]+=1
            if d.get('class_support'):
                assert d['intent']['forward']==0 and d['intent']['roll']==0
            checks.append(dict(t=source-start,frame_id=frame_id,old_accepted=old.get('accepted'),
                accepted=d['accepted'],old_state=old['state'],state=d['state'],
                association=d['tracking']['association_reason'],candidate_hits=d['tracking']['candidate_hits'],
                class_support=d['class_support'],measurement_id=d['measurement_id']))
    finally:capture.release()
    report=dict(scope='saved detections and pixels; fixed camera path, no flight/accuracy validation',
                original=dict(old_counts),replay=dict(counts),states=dict(states),checks=checks)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();replay(args.run,args.output)
