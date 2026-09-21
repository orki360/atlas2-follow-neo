"""Portable ONNX compute selection. No CUDA, TensorRT or vendor SDK imports.

DirectML uses DXGI adapter indices. CoreML permits CPU+GPU; macOS ultimately
chooses the hardware for CoreML partitions. Profiling verifies EP assignment,
not Apple's internal hardware scheduling.
"""
from collections import Counter
from pathlib import Path
import json
import platform
import time
import uuid
import numpy as np


def windows_adapters():
    """Enumerate DXGI in precisely the index order used by DirectML."""
    if platform.system() != 'Windows': return []
    import ctypes as c
    from ctypes import wintypes as w
    class GUID(c.Structure):
        _fields_ = [('a',w.DWORD),('b',w.WORD),('d',w.WORD),('e',c.c_ubyte*8)]
    class LUID(c.Structure):
        _fields_ = [('low',w.DWORD),('high',w.LONG)]
    class DESC(c.Structure):
        _fields_ = [('name',w.WCHAR*128),('vendor',w.UINT),('device',w.UINT),
                    ('subsys',w.UINT),('revision',w.UINT),('vram',c.c_size_t),
                    ('system',c.c_size_t),('shared',c.c_size_t),('luid',LUID),('flags',w.UINT)]
    def method(obj,index,restype,*args):
        table=c.cast(obj,c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(restype,c.c_void_p,*args)(table[index])
    factory=c.c_void_p(); adapters=[]
    dll=c.WinDLL('dxgi.dll')
    create=dll.CreateDXGIFactory1
    create.argtypes=[c.POINTER(GUID),c.POINTER(c.c_void_p)]; create.restype=c.c_long
    iid=GUID.from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    result=create(c.byref(iid),c.byref(factory))
    if result<0: raise RuntimeError(f'DXGI factory failed: {result}')
    try:
        for index in range(32):
            adapter=c.c_void_p()
            hr=method(factory,12,c.c_long,w.UINT,c.POINTER(c.c_void_p))(factory,index,c.byref(adapter))
            if hr & 0xffffffff == 0x887a0002: break
            if hr<0: raise RuntimeError(f'DXGI enumeration failed: {hr}')
            try:
                desc=DESC()
                hr=method(adapter,10,c.c_long,c.POINTER(DESC))(adapter,c.byref(desc))
                if hr<0: raise RuntimeError(f'DXGI description failed: {hr}')
                if not desc.flags & 2:
                    adapters.append({'id':index,'name':desc.name,'dedicated_memory_mb':desc.vram//1048576})
            finally: method(adapter,2,w.ULONG)(adapter)
    finally: method(factory,2,w.ULONG)(factory)
    return adapters


def provider_spec(system, available, device_id=0):
    if system=='Windows' and 'DmlExecutionProvider' in available:
        return ('DmlExecutionProvider',{'device_id':str(device_id)})
    if system=='Darwin' and 'CoreMLExecutionProvider' in available:
        # The pinned macOS wheels require macOS 13+, which supports MLProgram.
        return ('CoreMLExecutionProvider',{'ModelFormat':'MLProgram',
                'MLComputeUnits':'CPUAndGPU','RequireStaticInputShapes':'1'})
    return None


def create_session(model_path, mode='Auto', device_id=-1, diagnostics_dir=None):
    import onnxruntime as ort
    mode=mode.upper()
    if mode not in ('AUTO','GPU','CPU'): raise ValueError('Compute must be Auto, GPU or CPU')
    system=platform.system(); available=ort.get_available_providers()
    adapters=[]; adapter_warning=None
    if system=='Windows' and mode!='CPU':
        try: adapters=windows_adapters()
        except Exception as exc: adapter_warning=str(exc)
    if device_id<0:
        device_id=max(adapters,key=lambda a:a['dedicated_memory_mb'])['id'] if adapters else 0
    spec=provider_spec(system,available,device_id)
    info={'requested':mode,'platform':system,'available_providers':available,
          'adapters':adapters,'device_id':device_id,'adapter_warning':adapter_warning,
          'fallback_reason':None,'profile_node_counts':{}}

    def start(selected):
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern=selected is None or selected[0]!='DmlExecutionProvider'
        if selected and diagnostics_dir:
            Path(diagnostics_dir).mkdir(parents=True,exist_ok=True)
            options.enable_profiling=True
            options.profile_file_prefix=str(Path(diagnostics_dir)/('compute_'+uuid.uuid4().hex[:8]))
        providers=[selected,'CPUExecutionProvider'] if selected else ['CPUExecutionProvider']
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=providers)
        session.disable_fallback()
        if selected and selected[0] not in session.get_providers():
            raise RuntimeError(f'{selected[0]} did not initialize. Available session: {session.get_providers()}')
        inp=session.get_inputs()[0]
        sample=np.zeros([int(x) for x in inp.shape],np.float32)
        started=time.perf_counter()
        session.run(None,{inp.name:sample})
        warmup=(time.perf_counter()-started)*1000
        counts={}; profile=None
        if selected and diagnostics_dir:
            profile=session.end_profiling()
            events=json.loads(Path(profile).read_text(encoding='utf-8'))
            counts=dict(Counter(e.get('args',{}).get('provider') for e in events
                                if e.get('cat')=='Node' and e.get('args',{}).get('provider')))
            if not counts.get(selected[0]):
                raise RuntimeError(f'No model nodes executed by {selected[0]}; see {profile}')
        label='CPU'
        if selected:
            label='GPU / DirectML' if system=='Windows' else 'CoreML / CPU + GPU permitted'
        info.update(providers=session.get_providers(),label=label,warmup_ms=warmup,
                    profile_node_counts=counts,profile_path=profile,
                    gpu_hardware_execution_verified=bool(selected and system=='Windows' and counts.get(selected[0])))
        return session

    if mode=='CPU': return start(None),info
    try:
        if spec is None:
            raise RuntimeError(f'No compatible GPU provider for {system}. Available: {available}. Run SETUP again.')
        if spec[0]=='CoreMLExecutionProvider' and tuple(map(int,ort.__version__.split('.')[:2]))<(1,22):
            raise RuntimeError('CoreML options require ONNX Runtime 1.22+. Run SETUP again.')
        return start(spec),info
    except Exception as exc:
        if mode=='GPU':
            raise RuntimeError(f'GPU initialization failed: {exc}. Check the graphics driver or select CPU/Auto.') from exc
        info['fallback_reason']=str(exc)
        return start(None),info
