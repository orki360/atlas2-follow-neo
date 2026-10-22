"""Compare registered models on identical local video frames, without aircraft I/O."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np
from follow_neo.perception.detector import DetectorSettings, OnnxDroneDetector
from follow_neo.perception.models import DEFAULT_MODEL, HIGH_RES_MODEL, MODEL_NAMES, resolve_model


def load_frames(path, times):
    # Sequential decoding avoids repeated long-GOP seeks and selects identical
    # source pixels for every model and benchmark round.
    times = sorted(set(times))
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        origin = stream.start_time or 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float((frame.pts-origin) * stream.time_base)
            if actual + 1e-6 < times[len(frames)]:
                continue
            image = frame.to_ndarray(format='bgr24')
            while len(frames)<len(times) and actual+1e-6>=times[len(frames)]:
                frames.append((actual,image))
            if len(frames)==len(times):
                break
    if len(frames)!=len(times):
        raise RuntimeError(f'Cannot decode frame at {times[len(frames)]}s')
    return frames


def save_preview(image, detections, label, seconds, destination, confidence=.10):
    canvas = image.copy()
    for d in detections:
        if d['confidence']<confidence:
            continue
        color = (60,220,60) if d['class_id']==0 else (240,70,210)
        x1,y1,x2,y2 = [int(round(v)) for v in d['box']]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        for polygon in d.get('segments',[]):
            cv2.polylines(canvas,[np.asarray(polygon,np.int32)],True,color,1)
        cv2.putText(canvas,f"{d['class_name']} {d['confidence']:.2f}",
                    (x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    cv2.putText(canvas,f'{label} | video {seconds:.3f}s | conf >= {confidence:.2f}',
                (15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    if not cv2.imwrite(str(destination),canvas):
        raise RuntimeError(f'Cannot save {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--times',type=float,nargs='+',default=[15,40,59,75,98,100,101,120,137,145,165])
    parser.add_argument('--compute',choices=('GPU','CPU','Auto'),default='GPU')
    parser.add_argument('--models',choices=MODEL_NAMES,nargs='+',default=[DEFAULT_MODEL,HIGH_RES_MODEL])
    parser.add_argument('--confidence',type=float,default=.10)
    parser.add_argument('--preview-confidence',type=float,default=.65)
    args = parser.parse_args()
    if not 1<=args.rounds<=100 or any(not np.isfinite(t) or t<0 for t in args.times):
        parser.error('rounds must be 1..100 and times must be nonnegative')
    if not 0<args.confidence<=args.preview_confidence<=1:
        parser.error('require 0 < confidence <= preview-confidence <= 1')
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    print('Reading local video frames:',args.video,flush=True)
    frames = load_frames(args.video,args.times)
    report = dict(scope='offline detector only; no UI, tracking, recording or aircraft connection',
        accuracy_note='No ground-truth labels; confidence/counts are not accuracy metrics. Check model provenance before attributing differences to input resolution.',
        video=str(args.video.resolve()),rounds=args.rounds,confidence=args.confidence,
        preview_confidence=args.preview_confidence,opencv_threads=cv2.getNumThreads(),
        frames=[dict(seconds=t,sha256=hashlib.sha256(f.tobytes()).hexdigest()) for t,f in frames],models={})
    for name in dict.fromkeys(args.models):
        print('Loading model:',name,flush=True)
        destination = args.output/Path(name).stem
        destination.mkdir(exist_ok=True)
        path, manifest = resolve_model(ROOT,name)
        started = time.perf_counter()
        d = OnnxDroneDetector(path,DetectorSettings(confidence=args.confidence),compute_mode=args.compute,
                              diagnostics_dir=destination)
        load_ms = (time.perf_counter()-started)*1000
        for _,frame in frames[:5]:
            d.detect(frame)
        timings = []
        predictions = []
        for round_id in range(args.rounds):
            for i,(seconds,frame) in enumerate(frames):
                result = d.detect(frame)
                timings.append([result[key] for key in ('preprocess_ms','inference_ms','postprocess_ms','total_ms')])
                if round_id==0:
                    predictions.append(dict(seconds=seconds,frame_size=[frame.shape[1],frame.shape[0]],
                        input_transform=result['input_transform'],detections=result['detections']))
                    save_preview(frame,result['detections'],name,seconds,destination/f'frame_{i:02d}_{seconds:.3f}.jpg',args.preview_confidence)
            print(f'{name}: round {round_id+1}/{args.rounds} complete',flush=True)
        values = np.asarray(timings)
        summary = dict(model=manifest,input=d.input.shape,outputs=[o.shape for o in d.outputs],
            compute=d.compute_info,load_ms=load_ms,samples=len(timings),
            detector_fps=1000/float(values[:,3].mean()),
            inference_fps=1000/float(values[:,1].mean()),
            round_fps=[1000/float(v[:,3].mean()) for v in np.split(values,args.rounds)],
            timings_ms={key:dict(mean=float(values[:,i].mean()), **{label:float(np.percentile(values[:,i],p)) for label,p in (('median',50),('p95',95))})
                for i,key in enumerate(('preprocess','inference','postprocess','total'))},
            predictions=predictions)
        report['models'][name] = summary
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({name:{k:v for k,v in summary.items() if k not in ('predictions','model','compute')}}),flush=True)
        del d
        gc.collect()
    print('Report:',args.output/'comparison.json',flush=True)


if __name__=='__main__':
    main()
