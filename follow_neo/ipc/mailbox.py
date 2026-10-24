"""Latest-only snapshots. No shared object is ever flight authority.

Locks are nonblocking for bulk publication/reads: a slow observer drops a
preview, never stalls capture or control. Frame tokens prevent metadata from
being paired with an overwritten image. RawArray lifetime follows its owners.
"""
import pickle
import time


class Mailbox:
    def __init__(self, ctx, capacity=1024*1024):
        self.data=ctx.RawArray('B',capacity)
        self.length=ctx.RawValue('i',0)
        self.version=ctx.RawValue('q',0)
        self.lock=ctx.Lock()

    def put(self,value):
        raw=pickle.dumps(value,protocol=5)
        if len(raw)>len(self.data):raise ValueError('IPC snapshot exceeds bounded capacity')
        if not self.lock.acquire(False):return False
        try:
            memoryview(self.data).cast('B')[:len(raw)]=raw
            self.length.value=len(raw);self.version.value+=1
        finally:self.lock.release()
        return True

    def get(self,version=-1):
        if not self.lock.acquire(False):return None,version
        try:
            if self.version.value==version or not self.length.value:return None,version
            raw=bytes(memoryview(self.data).cast('B')[:self.length.value])
            current=self.version.value
        finally:self.lock.release()
        return pickle.loads(raw),current


class FrameSlot:
    def __init__(self,ctx,capacity=1920*1080*3):
        self.data=ctx.RawArray('B',capacity)
        self.token=ctx.RawValue('q',0)
        self.lock=ctx.Lock()

    def put(self,frame):
        import numpy as np
        pixels=np.asarray(frame.image)
        if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:
            raise ValueError('Preview requires uint8 BGR')
        if pixels.nbytes>len(self.data):return None
        if not self.lock.acquire(False):return None
        try:
            np.frombuffer(self.data,dtype=np.uint8,count=pixels.size)[:]=pixels.reshape(-1)
            self.token.value+=1
            return dict(token=self.token.value,shape=pixels.shape,frame_id=frame.frame_id,
                        decoded_at=frame.decoded_at,stream_epoch=frame.stream_epoch)
        finally:self.lock.release()

    def get(self,meta):
        import numpy as np
        from follow_neo.capture.video import Frame
        if not meta or not self.lock.acquire(False):return None
        try:
            if meta['token']!=self.token.value:return None
            image=np.frombuffer(self.data,dtype=np.uint8,count=int(np.prod(meta['shape']))).reshape(meta['shape']).copy()
        finally:self.lock.release()
        return Frame(meta['frame_id'],meta['decoded_at'],image,meta['stream_epoch'])


class InputState:
    """GUI-origin timestamp and release fence; polling cannot renew input.

    The shared release fence bypasses queued lifecycle operations. An enable
    issued before a later release can never restore authority afterwards.
    """
    def __init__(self,ctx):
        self.latest=Mailbox(ctx,4096)
        self.fence=ctx.Value('q',0)
        self.hard_abort=ctx.Event()

    def release(self):
        with self.fence.get_lock():
            self.fence.value+=1
            return self.fence.value

    def pulse(self,keys,identity,origin='gui'):
        return self.latest.put(dict(time=time.monotonic(),keys=tuple(keys),identity=identity,
                                    fence=self.fence.value,origin=origin))
