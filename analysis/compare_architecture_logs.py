"""Compare inline/process scheduling without conflating FPS with latency.

Read-only input. Writes no configuration and opens no network connections.
Usage: python analysis/compare_architecture_logs.py BEFORE_DIR AFTER_DIR
"""
import argparse
import json
from pathlib import Path


def summarize(folder):
    path=Path(folder)
    if path.is_dir():path=path/'events.jsonl'
    windows={};health=[];configs=[];faults={};modes=set();session=None;axes=[];times=[]
    for line in path.open(encoding='utf-8'):
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        event=row.get('event');times.append(row.get('monotonic',0))
        if event=='session_start':session=row
        if event in ('settings','search_settings','axis_speed_changed','real_time_mode'):
            configs.append(dict(event=event,time=row.get('monotonic'),settings=row.get('settings'),
                                axis_limits=row.get('axis_limits'),real_time=row.get('enabled')))
        if event=='architecture_configuration':axes.append(row.get('axis_limits'))
        if event in ('control_error','control_paused','control_movement_paused','video_error','controller_error'):
            key=event+': '+str(row.get('reason',row.get('error','')));faults[key]=faults.get(key,0)+1
        if event=='runtime_health':
            health.append(row);modes.add(str(row.get('architecture_mode','legacy')))
            for key,value in row.get('scheduling',{}).items():windows.setdefault(key,[]).append(value)
        if event=='architecture_health':
            for key,value in row.get('timing',{}).items():windows.setdefault(key,[]).append(value)
            for key,value in row.get('gui',{}).get('timing',{}).items():windows.setdefault('gui_'+key,[]).append(value)
    timing={}
    for key,items in windows.items():
        count=sum(i['count'] for i in items)
        timing[key]=dict(samples=count,mean=sum(i['mean']*i['count'] for i in items)/count,
                         maximum=max(i['maximum'] for i in items),max_window_p95=max(i['p95'] for i in items))
    rates={}
    for key in ('input_fps','processed_fps','inference_fps','result_age_ms','measurement_age_ms'):
        values=[r[key] for r in health if isinstance(r.get(key),(float,int))]
        if values:rates[key]=dict(mean=sum(values)/len(values),maximum=max(values),samples=len(values))
    stats=[r.get('command_statistics',{}) for r in health]
    commands=sum(r.get('commands_acknowledged',0) for r in stats)
    nonzero=sum(r.get('nonzero_commands',0) for r in stats)
    return dict(path=str(path.resolve()),architecture=sorted(modes),
        duration_seconds=max(times)-min(times) if times else 0,
        version=(session or {}).get('version'),settings=(session or {}).get('settings'),
        model=(session or {}).get('model'),initial_axis_limits=axes[0] if axes else None,
        real_time_at_start=(session or {}).get('real_time'),configuration_changes=configs,
        timing=timing,health_samples=rates,faults=faults,
        commands_in_health_windows=commands,nonzero_commands_in_health_windows=nonzero,
        ack_ms_max=max((r.get('ack_ms_max') or 0 for r in stats),default=0),
        command_gap_ms_max=max((r.get('command_gap_ms_max') or 0 for r in stats),default=0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('before');ap.add_argument('after');ap.add_argument('--output')
    args=ap.parse_args();before=summarize(args.before);after=summarize(args.after)
    same=all(before[k]==after[k] for k in ('settings','model','initial_axis_limits','real_time_at_start'))
    out=dict(before=before,after=after,initial_configuration_equal=same,
        interpretation='max_window_p95 is the largest 1-second window p95, not a global percentile. '
        'Nonzero command counts are not an aircraft trajectory measurement. '
        'Compare matching flight phases/settings; mixed Real-Time or tuning runs are not a controlled A/B test.')
    text=json.dumps(out,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
