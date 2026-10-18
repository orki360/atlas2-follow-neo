"""Offline detector benchmark on local video; never creates aircraft sockets."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from follow_neo.detector import OnnxDroneDetector, DetectorSettings
from follow_neo.models import DEFAULT_MODEL, resolve_model


def stats(values):
    return {key: float(np.percentile(values, p)) for key, p in
            (('median', 50), ('p95', 95), ('max', 100))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--inspect-graph', type=Path, help='Directory containing isolated ONNX analysis tools')
    ap.add_argument('--variants-tools', type=Path, help='Benchmark experimental pruned ONNX copies')
    args = ap.parse_args()
    if not 1 <= args.rounds <= 100:
        ap.error('rounds must be 1..100')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inspect_graph:
        sys.path.insert(0, str(args.inspect_graph))
        import onnx
        model, _ = resolve_model(ROOT, DEFAULT_MODEL)
        graph = onnx.load(str(model)).graph
        print(json.dumps(dict(outputs=[str(o) for o in graph.output],
            tail=[dict(name=n.name, op=n.op_type, inputs=list(n.input), outputs=list(n.output))
                  for n in graph.node[-45:]]), indent=2), flush=True)
        return
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError('Cannot open local video')
    frames = []
    try:
        for seconds in (15., 40., 59., 84., 90., 115.):
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot read video at {seconds}s')
            frames.append(frame)
    finally:
        capture.release()
    model, manifest = resolve_model(ROOT, DEFAULT_MODEL)
    detector = OnnxDroneDetector(model, DetectorSettings(confidence=.10),
                                 compute_mode='GPU', diagnostics_dir=args.output)
    for frame in frames:
        detector.detect(frame)
    if args.variants_tools:
        sys.path.insert(0, str(args.variants_tools))
        from benchmark_onnx_variants import benchmark_variants
        benchmark_variants(detector, frames, args.output, args.rounds)
        return
    report = dict(scope='offline detector only; no UI, recording or aircraft connection',
                  video=str(args.video), model=manifest, compute=detector.compute_info,
                  opencv_threads=cv2.getNumThreads(), modes={})
    references = []
    # Compare the same pixels/weights; requesting only the BBOX output is an
    # experiment, not a change to the model's graph or the application default.
    for mode in ('full', 'no_mask_rendering', 'boxes_output_only'):
        timings = []; boxes = []; errors = []
        for round_id in range(args.rounds):
            for i, frame in enumerate(frames):
                start = time.perf_counter()
                tensor, transform = detector._prepare(frame)
                prepared = time.perf_counter()
                names = [o.name for o in detector.outputs]
                outputs = detector.session.run(names[:1] if mode == 'boxes_output_only' else names,
                                               {detector.input.name: tensor})
                inferred = time.perf_counter()
                detections = detector.postprocess(outputs[0], transform,
                    outputs[1] if mode == 'full' and detector.segmentation else None)
                finished = time.perf_counter()
                timings.append(((prepared-start)*1000, (inferred-prepared)*1000,
                                (finished-inferred)*1000, (finished-start)*1000))
                plain = [{k: v for k, v in d.items() if k != 'segments'} for d in detections]
                if round_id == 0:
                    boxes.append(plain)
                if mode != 'full' and plain != references[i]:
                    errors.append(dict(round=round_id, frame=i, reason='BBOX/class output differs'))
        if mode == 'full':
            references = boxes
        values = np.asarray(timings)
        result = dict(samples=len(values), cycle_fps=1000/float(values[:, 3].mean()),
                      timings_ms={name: stats(values[:, i]) for i, name in enumerate(
                          ('preprocess', 'inference', 'postprocess', 'total'))},
                      equivalent_bbox_outputs=not errors, differences=errors[:10])
        report['modes'][mode] = result
        print(json.dumps({mode: result}), flush=True)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Report:', args.output/'benchmark.json', flush=True)


if __name__ == '__main__':
    main()
