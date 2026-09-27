"""Offline concurrent detector + paired-recording benchmark. No aircraft sockets.

Default input is a moving synthetic 1080p scene. --video uses a local video as
input (looped at 30 submitted frames/s); this is a load test, not an accuracy test.
"""
import argparse,json,time,threading
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from follow_neo.detector import OnnxDroneDetector
from follow_neo.models import resolve_model,DEFAULT_MODEL
from follow_neo.recording import VideoRecorder
from follow_neo.video import Frame,LatestValue
from follow_neo.controller import FollowController
from follow_neo.types import Settings,Box,Detection


def benchmark(seconds=130,compute='Auto',profile='Auto',model=DEFAULT_MODEL,video=None,output=None):
    root=Path(__file__).resolve().parent
    out=Path(output) if output else root/'logs'/('benchmark10_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    model_path,manifest=resolve_model(root,model)
    cv2.setNumThreads(2)
    detector=OnnxDroneDetector(model_path,compute_mode=compute,diagnostics_dir=out)
    frames=LatestValue();decisions=LatestValue();stop=threading.Event();errors=[];timings=[]
    settings=Settings();controller=FollowController()
    def infer():
        version=0
        try:
            while not stop.is_set():
                frame,new=frames.wait_next(version,.1)
                if new==version or frame is None:continue
                version=new;result=detector.detect(frame.image);now=time.monotonic()
                timings.append((now,result['inference_ms'],result['total_ms']))
                controller.observe([Detection(Box(*d['box']),d['confidence']) for d in result['detections']],
                    frame.decoded_at,now,frame.frame_id,(1920,1080),settings)
                decisions.set(controller.tick(now,1920,1080,frame.decoded_at,settings))
        except Exception as exc:errors.append(str(exc));stop.set()
    recorder=VideoRecorder(out,'Both videos',profile=profile)
    worker=threading.Thread(target=infer,daemon=True);worker.start()
    capture=cv2.VideoCapture(str(video)) if video else None
    if capture is not None and not capture.isOpened():
        stop.set();worker.join();recorder.wait();raise RuntimeError('Cannot read benchmark video')
    base=np.random.default_rng(42).integers(0,255,(270,480,3),dtype=np.uint8)
    base=cv2.resize(base,(1920,1080));start=time.monotonic();count=0
    try:
        while time.monotonic()-start<seconds and not stop.is_set():
            delay=start+count/30-time.monotonic()
            if delay>0:stop.wait(delay)
            if stop.is_set():break
            if capture is not None:
                ok,image=capture.read()
                if not ok:capture.set(cv2.CAP_PROP_POS_FRAMES,0);ok,image=capture.read()
                if not ok:raise RuntimeError('Cannot loop benchmark video')
                image=cv2.resize(image,(1920,1080))
            else:
                image=np.roll(base,count*7%1920,axis=1)
                cv2.rectangle(image,(count*9%1600,400),(count*9%1600+150,470),(230,230,230),-1)
            count+=1;frame=Frame(count,time.monotonic(),image)
            frames.set(frame);recorder.submit(frame,decisions.get())
            if count%300==0:print(f'{count/30:.0f}s | YOLO frames {len(timings)} | Recording {recorder.status()}',flush=True)
    finally:
        elapsed=time.monotonic()-start;stop.set();worker.join();recorder.wait()
        if capture is not None:capture.release()
    status=recorder.status()
    report=dict(scope='offline load test; no aircraft connection',model=manifest,compute=detector.compute_info,
        source=str(video) if video else 'moving synthetic 1080p',seconds=elapsed,submitted=count,
        submitted_fps=count/elapsed,inference_count=len(timings),inference_fps=len(timings)/elapsed,
        inference_ms_median=float(np.median([x[1] for x in timings])) if timings else None,
        pipeline_ms_p95=float(np.percentile([x[2] for x in timings],95)) if timings else None,
        recording=status,errors=errors,flight_commands_enabled=False,
        meets_30fps_target=bool(not errors and not recorder.error and len(timings)/elapsed>=28.5 and status['recorded_fps']>=28.5))
    (out/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=130)
    ap.add_argument('--compute',choices=['Auto','CPU','GPU'],default='Auto')
    ap.add_argument('--profile',choices=['Auto','1080p','720p'],default='Auto')
    ap.add_argument('--model',choices=[DEFAULT_MODEL,'best.onnx'],default=DEFAULT_MODEL)
    ap.add_argument('--video',type=Path);ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if not 2<=a.seconds<=600:ap.error('seconds must be between 2 and 600')
    r=benchmark(a.seconds,a.compute,a.profile,a.model,a.video,a.output)
    raise SystemExit(1 if r['errors'] or r['recording']['error'] else 0)
