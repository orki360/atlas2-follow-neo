"""Export the improved PT to a new fixed FP32 ONNX, in an isolated CPU environment.

Requires ultralytics==8.4.175, torch==2.7.1+cpu, torchvision==0.22.1+cpu,
onnx==1.17.0, onnxruntime==1.20.1 and numpy==2.0.2. Never install these
export-only dependencies into the application's DirectML environment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--validation-video', type=Path, required=True)
    args = parser.parse_args()
    source, output, report = (p.resolve() for p in (args.source,args.output,args.report))
    if source.suffix!='.pt' or not source.is_file():
        parser.error('source must be an existing local PT checkpoint')
    if output.exists() or output.suffix!='.onnx':
        parser.error('output must be a NEW .onnx file; originals will not be overwritten')
    report.parent.mkdir(parents=True,exist_ok=True)
    config_dir = report.parent/'ultralytics_settings'
    (config_dir/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO, __version__
    if __version__!='8.4.175':
        raise RuntimeError('Use Ultralytics 8.4.175 to match the source export semantics')
    torch.set_num_threads(2)
    originals = {p.name:sha256(p) for p in source.parent.iterdir() if p.suffix in ('.pt','.onnx')}
    exported_models = []
    with tempfile.TemporaryDirectory(prefix='export_960_',dir=report.parent) as folder:
        # Ultralytics writes beside its checkpoint. Work on a staging copy so
        # the original Full HD ONNX cannot be replaced by the smaller export.
        staged_pt = Path(folder)/(output.stem+'.pt')
        shutil.copy2(source,staged_pt)
        yolo = YOLO(str(staged_pt),task='segment')
        yolo.add_callback('on_export_end', lambda exporter: exported_models.append(exporter.model))
        settings = dict(format='onnx',imgsz=(544,960),batch=1,device='cpu',
                        quantize=32,dynamic=False,simplify=False,opset=17,nms=None)
        exported = Path(yolo.export(**settings))
        graph = onnx.load(str(exported))
        onnx.checker.check_model(graph,full_check=True)
        shape = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
        inputs = [shape(v) for v in graph.graph.input]
        outputs = [shape(v) for v in graph.graph.output]
        assert inputs==[[1,3,544,960]], inputs
        assert outputs==[[1,38,10710],[1,32,136,240]], outputs
        assert all(v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT
                   for v in [*graph.graph.input,*graph.graph.output])
        assert all(v.data_type!=onnx.TensorProto.FLOAT16 for v in graph.graph.initializer)
        assert not any(n.op_type in ('QuantizeLinear','DequantizeLinear') for n in graph.graph.node)
        metadata = {p.key:p.value for p in graph.metadata_props}
        assert metadata['end2end']=='False', metadata
        assert metadata['names']=="{0: 'NEO', 1: 'MINI PRO4'}", metadata

        # Compare identical pixels through the exact prepared PyTorch export
        # model and ONNX CPU. This checks conversion, not detection accuracy.
        capture = cv2.VideoCapture(str(args.validation_video.resolve()))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame.shape[:2]!=(1080,1920):
            raise RuntimeError('Expected a readable Full HD validation video')
        frame = cv2.copyMakeBorder(cv2.resize(frame,(960,540)),2,2,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor = np.ascontiguousarray(frame[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        with torch.inference_mode():
            reference = exported_models[0](torch.from_numpy(tensor))
        options = ort.SessionOptions()
        options.intra_op_num_threads=2
        session = ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
        actual = session.run(None,{'images':tensor})
        parity = []
        for index,(expected,result) in enumerate(zip(reference,actual,strict=True)):
            expected = expected.detach().cpu().numpy()
            np.testing.assert_allclose(result,expected,rtol=1e-4,atol=2e-3)
            parity.append(dict(output=index,max_absolute_error=float(np.max(np.abs(expected-result))),
                               mean_absolute_error=float(np.mean(np.abs(expected-result)))))
        assert all(sha256(source.parent/name)==value for name,value in originals.items())
        output.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation also protects a destination created during export.
        with output.open('xb') as target, exported.open('rb') as data:
            shutil.copyfileobj(data,target)
        result = dict(source=str(source),source_sha256=originals[source.name],output=str(output),
            sha256=sha256(output),bytes=output.stat().st_size,settings=settings,inputs=inputs,outputs=outputs,
            metadata=metadata,conversion_parity=parity,originals_unchanged=True,original_hashes=originals,
            versions={p:importlib.metadata.version(p) for p in ('torch','torchvision','ultralytics','onnx','onnxruntime','numpy')})
        report.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
