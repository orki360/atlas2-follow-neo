"""Read-only sampled bearing-controller comparison, never a flight replay.

HARD gaps contain no samples and are excluded, not filled or inferred. The
baseline source file is a local pre-change copy, supplied explicitly by path.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from follow_neo.control.centering import DampedTrackingController
from follow_neo.control.position import PositionSelector, select_position
from follow_neo.control.nano_guard import current_visual_support, sustained_real_neo
from follow_neo.shared.types import Settings, Kinematics


def distribution(items):
    items=sorted(items)
    return dict(count=len(items),mean=statistics.mean(items) if items else None,
                p95=items[min(len(items)-1,int(.95*len(items)))] if items else None,
                maximum=max(items) if items else None)


def analyze(path, baseline):
    stats={name:dict(positions=Counter(),switches=0,yaw_steps=[],switch_yaw_steps=[],
                    active_curves=0,limited=0,samples=0) for name in ('before','after')}
    controllers={'before':baseline(),'after':DampedTrackingController()}
    selector=PositionSelector();prior={};last=None;acks=[];gaps=[];commands=Counter();indicators=[]
    settings=Settings();recorded_sources=Counter();resets=0
    for line in path.open(encoding='utf-8'):
        r=json.loads(line);event=r['event']
        if last and r['monotonic']-last['monotonic']>2:
            gaps.append(dict(start_wall=last['wall_time'],end_wall=r['wall_time'],
                             seconds=r['monotonic']-last['monotonic']))
            controllers={'before':baseline(),'after':DampedTrackingController()}
            selector=PositionSelector();prior={};resets+=1
        last=r
        if 'settings' in r:
            settings=replace(settings,**{k:v for k,v in r['settings'].items() if k in Settings.__dataclass_fields__})
        if event=='control_restart':
            controllers={'before':baseline(),'after':DampedTrackingController()}
            selector=PositionSelector();prior={};resets+=1
        if event=='control_command_ack':acks.append(r['ack_ms'])
        if event=='flight_command':commands[r.get('selected_mode') or r.get('mode','unknown')]+=1
        if event=='control_indicator':indicators.append({k:r[k] for k in ('wall_time','title','detail')})
        if event!='observation':continue
        d=r['decision'];recorded_sources[(d.get('control_position') or {}).get('source','unknown')]+=1
        if not d.get('kinematics'):continue
        now=d['prediction_time'];width,height=d['frame_size'];k=Kinematics(**d['kinematics'])
        v=d.get('nano');position=d.get('control_position') or {}
        enabled=settings.nano_primary_position and position.get('reason') not in ('disabled','legacy_backend')
        age=float('inf') if d.get('measurement_time') is None else now-d['measurement_time']
        fresh=d.get('accepted',False);identity=(d.get('target_identity') or {}).get('logical_target_id')
        box=d.get('selected_measurement_box');merror=None if not box else ((box[0]+box[2])*.5-width*.5)/(width*.5)
        tracking=d.get('tracking') or {};warp=tracking.get('camera_warp')
        obs=None
        if fresh and identity is not None and d.get('class_support') is None and d.get('track_support')!='weak_yolo' and box:
            obs=dict(time=d['measurement_time'],point=((box[0]+box[2])*.5,(box[1]+box[3])*.5),
                     width=width,frame_id=d['measurement_id'],identity_key=(d.get('generation'),identity),
                     warp=warp,camera_valid=tracking.get('camera_motion')=='applied' and warp is not None)
        current=bool(obs and (d.get('tracking_confidence') or 0)>=.75
                     and current_visual_support(v,now,identity,d.get('generation'),list(vars(k.box).values())))
        # Confidence in a logged observation is decayed; this comparison is
        # intentionally sampled and does not claim to recreate every live tick.
        if (d.get('centering_control') or {}).get('combined_motion') and settings.forward_require_nano:
            current=True
        for name,ctl in controllers.items():
            chosen,info=(select_position(d,k,v,now,enabled) if name=='before'
                         else selector.select(d,k,v,now,enabled))
            out,_,_=ctl.compute(chosen,width,height,age,not fresh,now,settings,d.get('heading'),
                d.get('measurement_time'),merror,obs,current,sustained_real_neo(d),info)
            active=enabled and fresh and d['state']=='TRACK' and not d.get('stale')
            stat=stats[name];old=prior.get(name)
            if active:
                stat['samples']+=1;stat['positions'][info['source']]+=1
                stat['active_curves']+=int(ctl.diagnostics['early_curve']['active'])
                stat['limited']+=int(ctl.diagnostics.get('position_handoff_limited',False))
                if old and 0<now-old[0]<.15:
                    step=abs(out.yaw-old[2]);stat['yaw_steps'].append(step)
                    if info['source']!=old[1]:
                        stat['switches']+=1;stat['switch_yaw_steps'].append(step)
                prior[name]=(now,info['source'],out.yaw)
            else:prior.pop(name,None)
    for stat in stats.values():
        for key in ('yaw_steps','switch_yaw_steps'):stat[key]=distribution(stat[key])
    return dict(run=path.parent.name,scope='sampled centering only; no HARD data, images or aircraft dynamics',
                resets=resets,log_gaps=gaps,recorded_command_modes=commands,recorded_sources=recorded_sources,
                recorded_ack_ms=distribution(acks),indicators=indicators,comparison=stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--baseline-centering',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    module=types.ModuleType('follow_neo.control._baseline_centering')
    module.__package__='follow_neo.control'
    exec(compile(args.baseline_centering.read_text(encoding='utf-8'),str(args.baseline_centering),'exec'),module.__dict__)
    result=[analyze(p/'events.jsonl',module.DampedTrackingController) for p in args.run]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for item in result:print(json.dumps({k:item[k] for k in ('run','scope','comparison')},ensure_ascii=False))
