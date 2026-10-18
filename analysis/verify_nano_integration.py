"""Offline real-time replay of the actual inference/control/Nano workers.

Never calls FollowSession.start(), VideoReceiver.start() or ManualControl.
Network connection functions are blocked. Commands are calculated and logged,
never transmitted. Synthetic heading is explicitly labelled, not flight data.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from follow_neo.session import FollowSession
from follow_neo.types import settings_from_config
from follow_neo.dance import dance_command
from follow_neo.nano_guard import guard_reason
from follow_neo.overlay import render_bundle
import cv2


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95))) if values else {}


def run_case(name,start,end,output,config,video):
    settings=replace(settings_from_config(config),nano_mode='guarded',inference_fps=10.,video_fps=30.)
    session=FollowSession(ROOT,'OFFLINE-NO-NETWORK',settings=settings,
        compute_mode=config.get('compute','Auto'),device_id=config.get('gpu_adapter',-1),model_name=config['model_name'])
    session.receiver.state='STREAMING'
    session.control_provider=lambda:dict(active=True,mode='DANCE')
    session.heading.get=lambda:dict(yaw_deg=0.,time=time.monotonic(),source='offline_synthetic_heading')
    session.nano.start();session.inference_thread.start();session.control_thread.start()
    deadline=time.monotonic()+20
    while session.model_info is None and session.model_error is None and time.monotonic()<deadline:time.sleep(.01)
    if session.model_info is None:raise RuntimeError(session.model_error or 'model startup timeout')
    rows=[];started=None;cpu=None;count=0;first_source=None;last_frame_id=None
    try:
        with av.open(str(video)) as src:
            stream=src.streams.video[0];stream.codec_context.thread_count=2
            src.seek(int(start/float(stream.time_base)),stream=stream,backward=True)
            for f in src.decode(video=0):
                t=float(f.pts*f.time_base)
                if t<start:continue
                if t>end:break
                if first_source is None:
                    first_source=t;started=time.monotonic()+.05;cpu=time.process_time()
                stamp=started+t-first_source
                # Wait to the source timestamp, without queueing inference.
                delay=stamp-time.monotonic()
                if delay>0:time.sleep(delay)
                # Production timestamps frames at decoder publication, not at
                # camera exposure. Keep source PTS separately in row['t'].
                session.receiver._publish_decoded(f,time.monotonic())
                frame=session.receiver.frames.get()
                if frame is None or frame.frame_id==last_frame_id:continue
                last_frame_id=frame.frame_id;count+=1
                now=time.monotonic();d=session.control_decision()
                values,reason=dance_command(d,now,started,require_target_policy=True)
                v=(d or {}).get('nano') or {}
                rows.append(dict(t=t,wall_time=now,frame_id=frame.frame_id,decision=d,
                                 command=list(values),command_reason=reason))
                if (v.get('mode')=='guarded' and (d or {}).get('nano_authority') not in ('yolo_only','search')
                        and (guard_reason(v,now) or (d or {}).get('nano_authority')=='blocked')):
                    assert not any(values),'blocked visual result permitted movement'
                if d and d.get('nano_authority')=='bridge_yaw':
                    assert not any(values[1:]),'Nano-only bridge permitted translation/vertical'
                if count%30==0:
                    print(name,round(t,3),'YOLO',session.inference_count,'Nano',session.nano.updates,
                          'guard',v.get('reason'),flush=True)
                if name=='exit' and 198.5<t<198.55 or name=='blur' and 51.9<t<51.94:
                    rendered,_,_=render_bundle(frame,None,d,'Live prediction',now=now)
                    cv2.imwrite(str(output/f'{name}_preview.jpg'),rendered)
        wall=time.monotonic()-started
        cpu_seconds=time.process_time()-cpu
    finally:session.stop()
    events=[json.loads(line) for line in (session.output/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    errors=[r for r in events if r['event'] in ('nano_error','inference_error','controller_error')]
    observations=[r for r in events if r['event']=='observation']
    nanos=[r for r in events if r['event']=='nano_observation']
    visible=[r for r in rows if (r['decision'] or {}).get('nano_authority') in ('bridge_yaw','bridge_forward','verified_yaw')]
    absent=[r for r in rows if (name=='exit' and 198.422<=r['t']<=199.14)
            or (name=='blur' and 52.203<=r['t']<=52.515)]
    summary=dict(name=name,source_range=[start,end],frames=count,wall_seconds=wall,video_fps=count/wall,
        cpu_percent_machine=cpu_seconds/wall/os.cpu_count()*100,yolo_fps=session.inference_count/wall,
        nano_updates=session.nano.updates,nano_effective_fps=session.nano.updates/wall,
        supported_frames=len(visible),authority=Counter((r['decision'] or {}).get('nano_authority','none') for r in rows),
        nonzero_command_frames=sum(any(r['command']) for r in rows),
        nano_reinitializations=sum(r['event']=='nano_anchor_applied' for r in events),
        guard_reasons=Counter(((r['decision'] or {}).get('nano') or {}).get('reason','none') for r in rows),
        update_ms=stats([r['update_ms'] for r in nanos if 'update_ms' in r]),
        cycle_ms=stats([r['cycle_ms'] for r in nanos if 'cycle_ms' in r]),
        inference_ms=stats([r['inference_ms'] for r in observations if 'inference_ms' in r]),
        vetoes=[dict(reason=r['reason'],time=r['time']-started+first_source) for r in events if r['event']=='nano_veto'],
        absent_frames=len(absent),absent_nonzero_commands=sum(any(r['command']) for r in absent),
        absent_translation=sum(any(r['command'][1:]) for r in absent),
        absent_nano_motion=sum(any(r['command']) and (r['decision'] or {}).get('nano_authority') in ('bridge_yaw','bridge_forward') for r in absent),errors=errors,
        session_log=str(session.output/'events.jsonl'),
        method='Actual production workers, original video PTS paced in real time, target30 video/10 YOLO FPS. No sockets started. Synthetic stationary heading. Timing/control regression, not closed-loop flight or annotated tracking accuracy.')
    (output/f'{name}_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (output/f'{name}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary),flush=True)
    assert not errors,errors
    assert not summary['absent_translation'],'Known absent-target frames permitted translation'
    assert not summary['absent_nano_motion'],'Known absent-target frames permitted Nano target-following motion'
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case',default='all')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'config.local.json').read_text())
    video=ROOT/'logs/recording_171706_539dab_vrs13_teamOr_notMy/clean.mp4'
    cases=[('short_gap',5.,9.5),('blur',48.,52.7),('whiteboard',108.,114.5),('exit',193.,199.5)]
    def deny(*args,**kwargs):raise AssertionError('Network access forbidden in offline replay')
    with patch('socket.create_connection',deny),patch('socket.socket.connect',deny):
        summaries=[run_case(n,a,b,out,config,video) for n,a,b in cases if args.case in ('all',n)]
    (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__':main()
