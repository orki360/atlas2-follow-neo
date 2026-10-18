"""Named, checksum-verified local models. Never download or substitute weights."""
import hashlib
import json
from pathlib import Path

DEFAULT_MODEL='yolo26s_seg_best_mp4_v2.onnx'
MID_RES_MODEL='yolo26s_seg_best_mp4_960x544_v2_fp32.onnx'
V6_MODEL='neo_minipro4_yolo26s_seg_960x544_v6.onnx'
V6_FULL_AP_MODEL='neo_minipro4_yolo26s_seg_best_full_ap_960x544_v6.onnx'
HIGH_RES_MODEL='yolo26s_seg_best_mp4_1920x1080_v2.onnx'
PREVIOUS_MODEL='yolo26s_seg_best_v1.onnx'
LEGACY_MODEL='best.onnx'
MODEL_NAMES=(DEFAULT_MODEL,MID_RES_MODEL,V6_MODEL,V6_FULL_AP_MODEL,
             HIGH_RES_MODEL,PREVIOUS_MODEL,LEGACY_MODEL)
NEO_CLASS_ID=0
NEO_CLASS_NAME='neo'
MINI_CLASS_NAME='MINI PRO4'
DUAL_CLASS_NAMES={0:NEO_CLASS_NAME,1:MINI_CLASS_NAME}
DUAL_SOURCE_NAMES={0:'DJI-Neo-Segmentation-Manual',1:'minip4'}
DUAL_MODEL_SOURCE_NAMES={DEFAULT_MODEL:DUAL_SOURCE_NAMES,
                         MID_RES_MODEL:{0:'NEO',1:'MINI PRO4'},
                         V6_MODEL:{0:'NEO',1:'MINI PRO4'},
                         V6_FULL_AP_MODEL:{0:'NEO',1:'MINI PRO4'},
                         HIGH_RES_MODEL:{0:'NEO',1:'MINI PRO4'}}
DUAL_MODELS=tuple(DUAL_MODEL_SOURCE_NAMES)


def model_from_config(config):
    """Migrate once to v13's dual-class default, retaining other user settings."""
    if config.get('schema_version',1)<7:
        return DEFAULT_MODEL
    name=config.get('model_name',DEFAULT_MODEL)
    return name if name in MODEL_NAMES else DEFAULT_MODEL


def target_policy(model_name,source_names=None):
    """Require the selected registered model's exact NEO/MINI class mapping."""
    allowed=(model_name in DUAL_MODELS and source_names==DUAL_MODEL_SOURCE_NAMES[model_name])
    return dict(model_name=model_name,dance_allowed=allowed,
                target_class_id=NEO_CLASS_ID,target_class_name=NEO_CLASS_NAME,
                reason='neo_only' if allowed else 'Select the dual-class model for neo Dance')


def resolve_model(root,name=None):
    root=Path(root)
    name=name or DEFAULT_MODEL
    if name not in MODEL_NAMES:
        raise ValueError('Select one of the installed, verified models.')
    path=root/'models'/name
    registry=json.loads((root/'models'/'models_update10.json').read_text(encoding='utf-8'))
    expected=registry[name]['sha256']
    if not path.is_file():raise FileNotFoundError('Model missing: '+str(path))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('Model checksum mismatch: '+name)
    return path,dict(registry[name],name=name)
