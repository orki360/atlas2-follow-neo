"""Offline paired replay of logged detections and recorded camera frames. No I/O to aircraft."""
import argparse,json,sys,time
from pathlib import Path
from collections import Counter
import cv2
parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--logs',default='D:/projects/vers1_integration_yam/logs/20260928_125435_be0292')
args=parser.parse_args();sys.path.insert(0,str(Path(args.source).resolve()))
from follow_neo.controller import FollowController
from follow_neo.types import Settings,Detection,Box
from follow_neo.botsort_tracker import BoTSORTTracker
cv2.setNumThreads(2)
root=Path(args.logs)
events=[json.loads(line) for line in (root/'events.jsonl').open(encoding='utf-8')]
observations={e['result_frame_id']:e for e in events if e['event']=='observation' and 'result_frame_id' in e}
names=[p.name for p in sorted(root.glob('recording_*')) if (p/'clean.mp4').exists()]
reports=[]
for name in names:
    frames=[json.loads(line) for line in (root/name/'frames.jsonl').open()]
    settings=next(e['settings'] for e in reversed(events) if e['event'] in ('settings','session_start') and e['monotonic']<=frames[0]['decoded_at'])
    settings=Settings(**settings);c=FollowController(tracker_backend='botsort')
    cap=cv2.VideoCapture(str(root/name/'clean.mp4'));rows=[];states=Counter();seen=set();last_id=None;switches=0
    for f in frames:
        ok,image=cap.read()
        if not ok:raise RuntimeError('Video ended before metadata')
        e=observations.get(f['frame_id'])
        if e is None or e['result_frame_id'] in seen:continue
        seen.add(e['result_frame_id']);t=f['decoded_at'];now=e['decision']['prediction_time']
        ds=[Detection(Box(*d['box']),d['confidence']) for d in e['detections']]
        heading=e['decision'].get('heading')
        h,w=image.shape[:2]
        c.observe(ds,t,now,f['frame_id'],(w,h),settings,image=image)
        d=c.tick(now,w,h,t,settings,heading)
        tid=d['tracking']['target_id']
        if tid is not None and last_id is not None and tid!=last_id:switches+=1
        if tid is not None:last_id=tid
        states[d['state']]+=1
        rows.append(dict(frame=f['frame_id'],time=t,accepted=d['accepted'],state=d['state'],
            confidence=max([x.confidence for x in ds],default=0),id=tid,tracking=d['tracking'],
            box=d['track_box'],measurement_age_ms=d['measurement_age_ms'],candidate_pending=d.get('candidate_pending',False),
            centering_control=d.get('centering_control'),intent=d['intent'],edge_search=d.get('edge_search')))
    cap.release()
    report=dict(recording=name,matched_frames=len(rows),recorded_frames=len(frames),accepted=sum(r['accepted'] for r in rows),
                search=sum(r['state']=='DIRECTIONAL_SEARCH' for r in rows),id_changes=switches,
                strong_rejected=sum(r['confidence']>=.8 and not r['accepted'] for r in rows),states=dict(states))
    reports.append(report);print(json.dumps(report),flush=True)
    path=Path(args.output);path.with_name(path.stem+'_'+name+'.json').write_text(json.dumps(rows),encoding='utf-8')
Path(args.output).write_text(json.dumps(reports,indent=2),encoding='utf-8')
