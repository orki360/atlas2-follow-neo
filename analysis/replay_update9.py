"""Replay logged detections through Update 9. No sockets or flight simulation.

Without recorded heading, search eligibility can be evaluated, but the measured
angle controller must wait. The old automatic Dance-stop events do not terminate
this counterfactual replay. Aircraft response to new commands is not simulated.
"""
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_neo.controller import FollowController
from follow_neo.dance import dance_command
from follow_neo.types import Box,Detection,settings_from_config


def replay(path):
    c=FollowController();s=settings_from_config({});last_id=None;heading=None
    counts=collections.Counter();states=collections.Counter();spacing=collections.Counter()
    original_terminal=collections.Counter();reasons=collections.Counter();episodes=set()
    active=False;started=0.;generation=0;resumed_after_loss=0;lost=False
    for line in Path(path).open(encoding='utf-8'):
        e=json.loads(line);event=e['event']
        if event in ('session_start','settings'):s=settings_from_config({'schema_version':4,'settings':e['settings']})
        if event=='heading':heading={k:e[k] for k in ('time','yaw_deg','source')}
        if event=='dance_selection':active=e['selected'];started=e['monotonic']
        if event=='dance_terminal_stop':original_terminal[e['reason']]+=1
        if event=='tracking_reset':c.reset();last_id=None;generation+=1;lost=False
        if event=='recording_stop_requested':active=False
        if event!='observation':continue
        old=e['decision'];now=old['prediction_time'];size=tuple(old['frame_size'])
        fid=e.get('result_frame_id')
        if fid is not None and fid!=last_id:
            ds=[Detection(Box(*d['box']),d['confidence']) for d in e.get('detections',[])]
            c.observe(ds,e['decoded_at'],now,fid,size,s);last_id=fid
        d=c.tick(now,*size,now-s.stale_seconds-.01 if old['stale'] else now,s,heading)
        values,status=dance_command(d,now,started)
        # Serialization is part of the test: logging must not receive NaN/Inf.
        
        try:json.dumps(d,allow_nan=False)
        except TypeError:
            def inspect(o,path='decision'):
                if isinstance(o,dict):
                    for k,v in o.items():inspect(v,path+'.'+k)
                elif isinstance(o,(list,tuple)):
                    for i,v in enumerate(o):inspect(v,path+'.'+str(i))
                elif type(o).__module__!='builtins':print(path,type(o),repr(o))
            inspect(d);raise
        counts['observations']+=1
        if not active:continue
        counts['active_replay_observations']+=1;states[d['state']]+=1;spacing[d['spacing_phase']]+=1
        counts['forward_observations']+=int(values[3]>0)
        counts['predicted_forward_observations']+=int(d['prediction_forward_allowed'] and values[3]>0)
        counts['terminal_observations']+=int(d['spacing_phase'] in ('STOPPED','SEQUENCE_DONE'))
        counts['nonzero_yaw_search_observations']+=int(status.startswith('Search /') and values[0]!=0)
        search=d['edge_search'];reasons[search['reason']]+=1
        if search['reason']=='waiting_heading':episodes.add((generation,search['measurement_id']))
        if d['state'] in ('HOVER_WAIT','DIRECTIONAL_SEARCH','WAIT_VIDEO'):lost=True
        if lost and d['state']=='TRACK':resumed_after_loss+=1;lost=False
    return dict(scope='counterfactual observation replay, not physical flight validation',
        heading_logged=heading is not None,counts=dict(counts),states=dict(states),spacing=dict(spacing),
        original_terminal_events=dict(original_terminal),search_reasons=dict(reasons),
        eligible_search_episodes_waiting_heading=len(episodes),automatic_reacquisitions=resumed_after_loss)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('events',type=Path)
    ap.add_argument('--output',type=Path);args=ap.parse_args()
    report=json.dumps(replay(args.events),indent=2)
    if args.output:args.output.write_text(report,encoding='utf-8')
    print(report)
