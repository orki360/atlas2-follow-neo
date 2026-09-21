#!/usr/bin/env python3
"""Standalone lab and offline compute check, without any phone connection."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parent


def check(mode='Auto', device_id=-1):
    import numpy as np
    import cv2, av, onnxruntime, PIL, tkinter
    from follow_neo.detector import OnnxDroneDetector
    from follow_neo import __version__
    model=ROOT/'models'/'best.onnx';out=ROOT/'logs';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'models'/'model_info.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    if digest!=manifest['sha256']: raise RuntimeError('Model checksum mismatch. Reapply the update ZIP.')
    detector=OnnxDroneDetector(model,compute_mode=mode,device_id=device_id,diagnostics_dir=out)
    av.Codec('libx264','w')
    timings=[]
    for _ in range(3):
        result=detector.detect(np.zeros((1080,1920,3),np.uint8));timings.append(result['inference_ms'])
    report={'status':'PASS','version':__version__,'python':sys.version.split()[0],
      'numpy':np.__version__,'opencv':cv2.__version__,'av':av.__version__,
      'onnxruntime':onnxruntime.__version__,'pillow':PIL.__version__,
      'tk':tkinter.TkVersion,'model_sha256':digest,'input':detector.input.shape,
      'output':detector.output.shape,'compute':detector.compute_info,
      'recording_encoder':'libx264 available','source_resolution':[1920,1080],'blank_frame_detections':len(result['detections']),
      'warm_inference_ms':timings,'flight_commands_enabled':False}
    (out/'environment_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if detector.compute_info.get('fallback_reason'): print('WARNING: GPU unavailable; Auto is using CPU. See fallback_reason above.')


def main():
    ap=argparse.ArgumentParser(description='ATLAS2 Follow NEO: live video and perception preview only.')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--compute',choices=['Auto','GPU','CPU'],default='Auto')
    ap.add_argument('--device-id',type=int,default=-1)
    args=ap.parse_args()
    if args.check: check(args.compute,args.device_id);return
    import tkinter as tk
    from follow_neo.gui import FollowLabWindow
    root=tk.Tk();FollowLabWindow(root,ROOT);root.mainloop()


if __name__=='__main__':
    try: main()
    except Exception:
        details=traceback.format_exc();print(details,file=sys.stderr)
        (ROOT/'logs').mkdir(exist_ok=True)
        (ROOT/'logs'/'last_error.txt').write_text(details,encoding='utf-8');sys.exit(1)
