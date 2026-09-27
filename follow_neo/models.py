"""Named, checksum-verified local models. Never download or substitute weights."""
import hashlib
import json
from pathlib import Path

DEFAULT_MODEL='yolo26s_seg_best_v1.onnx'
LEGACY_MODEL='best.onnx'


def resolve_model(root,name=None):
    root=Path(root)
    name=name or DEFAULT_MODEL
    if name not in (DEFAULT_MODEL,LEGACY_MODEL):
        raise ValueError('Select one of the installed, verified models.')
    path=root/'models'/name
    registry=json.loads((root/'models'/'models_update10.json').read_text(encoding='utf-8'))
    expected=registry[name]['sha256']
    if not path.is_file():raise FileNotFoundError('Model missing: '+str(path))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('Model checksum mismatch: '+name)
    return path,dict(registry[name],name=name)
