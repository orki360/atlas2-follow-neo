"""Read-only authorization replay of saved decisions; no detector or aircraft I/O.

This isolates class-provenance loss. It does not simulate new video, camera
motion, the transport smoother, or the physical outcome of different commands.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from follow_neo.control.dance import dance_command
from follow_neo.perception.models import NEO_CLASS_ID,NEO_CLASS_NAME


def measurement_key(decision):
    return (decision.get('tracking',{}).get('logical_target_id'),
            decision.get('measurement_id'),decision.get('measurement_time'),
            tuple(decision.get('frame_size') or ()))


def verify_rows(rows):
    bound=None;started=None;checks=[];original_reasons=Counter();restored_reasons=Counter()
    for row in rows:
        event=row.get('event')
        if event in ('session_start','tracking_reset','tracking_numerical_reset'):
            bound=None
        if event=='control_mode' and row.get('mode')=='DANCE':started=row['monotonic']
        if event=='control_movement_resumed' and started is not None:started=row['monotonic']
        if event!='observation':continue
        decision=row.get('decision') or {};key=measurement_key(decision)
        if not decision.get('confirmed') or key[0] is None:bound=None
        elif (decision.get('accepted') and decision.get('selected_class_id')==NEO_CLASS_ID
              and decision.get('selected_class_name')==NEO_CLASS_NAME):
            bound=key
        elif bound is not None and key!=bound:bound=None
        # Only recover an exact previous *confirmed real measurement*, and
        # only evaluate when the recorded controller had Dance authority.
        authority=decision.get('control_status') or {}
        if (bound is None or key!=bound or started is None or decision.get('accepted')
            or decision.get('selected_class_id') is not None
            or not (decision.get('prediction_recovery') or decision.get('edge_search',{}).get('active'))
            or not authority.get('active') or authority.get('mode')!='DANCE'):
            continue
        now=decision['prediction_time']
        original,reason=dance_command(decision,now,started,require_target_policy=True)
        recovered=dict(decision,selected_class_id=NEO_CLASS_ID,selected_class_name=NEO_CLASS_NAME)
        restored,new_reason=dance_command(recovered,now,started,require_target_policy=True)
        original_reasons[reason]+=1;restored_reasons[new_reason]+=1
        checks.append(dict(time=now,frame_id=row.get('result_frame_id'),state=decision['state'],
            logical_target_id=key[0],measurement_id=key[1],measurement_time=key[2],
            original_reason=reason,original_values=original,restored_reason=new_reason,restored_values=restored))
    return dict(scope='saved-decision authorization only; proven neo measurement provenance; no flight validation',
        eligible_retained_identity_decisions=len(checks),
        original_nonzero=sum(any(r['original_values']) for r in checks),
        restored_nonzero=sum(any(r['restored_values']) for r in checks),
        states=dict(Counter(r['state'] for r in checks)),
        original_reasons=dict(original_reasons),restored_reasons=dict(restored_reasons),checks=checks)


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    # The input run is evidence. Never overwrite it, its videos or sidecars.
    if args.output and args.output.resolve().is_relative_to(args.events.resolve().parent):
        parser.error('Output must be outside the source run directory')
    report=verify_rows(read_rows(args.events))
    if args.output:args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2,ensure_ascii=False))
