"""Experimental copies only. Called by benchmark_inference, never by the app."""
import copy
import hashlib
import json
import time
import numpy as np
import onnx
from onnxconverter_common import float16
from follow_neo.compute import create_session


def prune_masks(model):
    result = copy.deepcopy(model)
    graph = result.graph
    if [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim] != [1, 38, 8400]:
        raise ValueError('This experiment requires the verified dual-class 640 segmentation graph')
    final = next(n for n in graph.node if graph.output[0].name in n.output)
    if final.op_type != 'Concat' or len(final.input) != 3:
        raise ValueError('Expected boxes, classes, coefficients concatenation')
    # The inspected graph concatenates [xywh, two class scores, mask coefficients].
    del final.input[2:]
    del graph.output[1:]
    graph.output[0].type.tensor_type.shape.dim[1].dim_value = 6
    producers = {name: n for n in graph.node for name in n.output}
    needed = set(); stack = [graph.output[0].name]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        if name in producers:
            stack.extend(producers[name].input)
    nodes = [n for n in graph.node if any(name in needed for name in n.output)]
    weights = [v for v in graph.initializer if v.name in needed]
    del graph.node[:]; graph.node.extend(nodes)
    del graph.initializer[:]; graph.initializer.extend(weights)
    del graph.value_info[:]
    onnx.checker.check_model(result)
    return result


def benchmark_variants(detector, frames, output, rounds):
    source = onnx.load(str(detector.model_path))
    pruned = prune_masks(source)
    refs = []
    for frame in frames:
        tensor, _ = detector._prepare(frame)
        refs.append(detector.session.run(None, {detector.input.name: tensor})[0][:, :6])
    reports = {}
    for name in ('bbox_fp32', 'bbox_fp16'):
        report = {}
        try:
            variant = pruned if name.endswith('fp32') else float16.convert_float_to_float16(
                copy.deepcopy(pruned), keep_io_types=True)
            path = output/(name+'.onnx')
            onnx.checker.check_model(variant)
            onnx.save(variant, str(path))
            session, info = create_session(path, 'GPU', diagnostics_dir=output)
            inp = session.get_inputs()[0].name
            for frame in frames:
                tensor, _ = detector._prepare(frame)
                session.run(None, {inp: tensor})
            timing = []; differences = []; class_changes = 0; output_changes = []
            for repeat in range(rounds):
                for i, frame in enumerate(frames):
                    start = time.perf_counter()
                    tensor, transform = detector._prepare(frame)
                    prepared = time.perf_counter()
                    raw = session.run(None, {inp: tensor})[0]
                    inferred = time.perf_counter()
                    detections = detector.postprocess(raw, transform)
                    finished = time.perf_counter()
                    timing.append([(prepared-start)*1000, (inferred-prepared)*1000,
                                   (finished-inferred)*1000, (finished-start)*1000])
                    if repeat == 0:
                        diff = np.abs(raw-refs[i])
                        differences.append(dict(frame=i, boxes_max_abs=float(diff[:, :4].max()),
                                                scores_max_abs=float(diff[:, 4:].max())))
                        high = np.maximum(raw[:, 4:].max(axis=1), refs[i][:, 4:].max(axis=1)) >= .1
                        class_changes += int(np.count_nonzero(
                            (raw[:, 4:].argmax(axis=1) != refs[i][:, 4:].argmax(axis=1)) & high))
                        expected = detector.postprocess(refs[i], transform)
                        output_changes.append(dict(frame=i, before=len(expected), after=len(detections),
                            before_classes=[d['class_id'] for d in expected],
                            after_classes=[d['class_id'] for d in detections]))
            values = np.asarray(timing)
            report = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(detector.model_path.read_bytes()).hexdigest(),
                source_nodes=len(source.graph.node), nodes=len(variant.graph.node), compute=info,
                cycle_fps=1000/float(values[:, 3].mean()), samples=len(values),
                timings_ms={key: dict(median=float(np.median(values[:, i])),
                    p95=float(np.percentile(values[:, i], 95))) for i, key in enumerate(
                        ('preprocess', 'inference', 'postprocess', 'total'))},
                raw_differences=differences, above_threshold_class_changes=class_changes,
                detection_counts=output_changes, activated_in_application=False)
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        reports[name] = report
        print(json.dumps({name: report}), flush=True)
        (output/'variants.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
