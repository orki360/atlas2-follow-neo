"""Offline sampler comparison. Video pacing is synthetic, never camera timing.

No sockets, inference or aircraft control. --video additionally exercises the
publication worker with real decoded frames and BGR conversion from a recording.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_neo.capture.cadence import FrameRateGate,LatestFrameSampler
from follow_neo.capture.video import VideoReceiver


def arrivals(kind,duration):
    if kind=='steady60':return [i/60 for i in range(int(duration*60))]
    if kind=='quantized60':return [round(i/60/.015625)*.015625 for i in range(int(duration*60))]
    rng=random.Random(321);out=[];t=0.
    shape=.5 if kind=='jitter' else 1/3
    while t<duration:
        out.append(t);t+=rng.gammavariate(shape,.019/shape)
    return out


def simulate(kind,duration=20):
    stamps=arrivals(kind,duration);old=FrameRateGate();before=sum(old.admit(t,30) for t in stamps)
    new=LatestFrameSampler();j=0;selected=[];times=[]
    for k in range(int(duration*1000)):
        now=k/1000
        while j<len(stamps) and stamps[j]<=now:new.offer(j,stamps[j]);j+=1
        value=new.take(now,30)
        if value is not None:selected.append((value,now-stamps[value]));times.append(now)
    return dict(case=kind,source_fps=len(stamps)/duration,old_fps=before/duration,
        new_fps=len(selected)/duration,unique=len(set(v for v,a in selected))==len(selected),
        age_mean_ms=statistics.mean(a for v,a in selected)*1000,
        age_max_ms=max(a for v,a in selected)*1000,
        publication_gap_max_ms=max(b-a for a,b in zip(times,times[1:]))*1000)


def video_trial(path,mode,duration=6,kind='steady60'):
    import av
    stamps=arrivals(kind,duration);published=[]
    def callback(frame):published.append((frame.frame_id,frame.decoded_at,time.monotonic()))
    receiver=VideoReceiver('offline.invalid',on_frame=callback)
    gate=FrameRateGate();old_count=0
    if mode=='new':
        receiver.publisher_thread=threading.Thread(target=receiver._publish_loop,daemon=True)
        receiver.publisher_thread.start()
    decode_ms=0.;cpu=time.process_time();start=time.perf_counter()
    try:
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            stream.codec_context.thread_count=2;stream.codec_context.thread_type='SLICE'
            frames=container.decode(stream)
            for i,due in enumerate(stamps):
                delay=start+due-time.perf_counter()
                if delay>0:time.sleep(delay)
                before=time.perf_counter();frame=next(frames)
                decode_ms+=(time.perf_counter()-before)*1000;source=time.monotonic()
                if mode=='new':receiver._publish_decoded(frame,source)
                elif gate.admit(source,30):
                    frame.to_ndarray(format='bgr24');old_count+=1
                    published.append((i+1,source,time.monotonic()))
            delay=start+duration-time.perf_counter()
            if delay>0:time.sleep(delay)
    finally:
        if mode=='new':receiver.stop()
    elapsed=time.perf_counter()-start
    return dict(mode=mode,case=kind,seconds=elapsed,decoded=len(stamps),published=len(published),
        published_fps=len(published)/elapsed,decode_mean_ms=decode_ms/len(stamps),
        cpu_core_equivalents=(time.process_time()-cpu)/elapsed,
        age_mean_ms=statistics.mean(t-s for i,s,t in published)*1000,
        age_max_ms=max(t-s for i,s,t in published)*1000,
        unique=len({i for i,s,t in published})==len(published),
        sampling=receiver.sampler.snapshot() if mode=='new' else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=dict(scope='offline synthetic arrival timing; no inference or live network; not a flight FPS claim',
                simulations=[simulate(k) for k in ('steady60','quantized60','jitter','bursty')])
    if args.video:
        report['video']=str(args.video.resolve())
        report['video_trials']=[video_trial(args.video,mode,kind=kind)
                                for kind in ('steady60','jitter') for mode in ('old','new')]
    value=json.dumps(report,indent=2,allow_nan=False)
    if args.output:args.output.write_text(value,encoding='utf-8')
    print(value)


if __name__=='__main__':main()
