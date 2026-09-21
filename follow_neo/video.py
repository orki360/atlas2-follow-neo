"""Receive-only MSDKRemote elementary H264/H265 TCP video (default 9999).

No control/query connection and no application bytes are ever sent to the phone.
This mirrors VideoServer.java: a raw elementary stream, no length prefix.
"""
from dataclasses import dataclass
import socket
import threading
import time
import av


@dataclass(frozen=True)
class Frame:
    frame_id: int
    decoded_at: float
    image: object


class LatestValue:
    def __init__(self): self.lock=threading.Lock(); self.value=None
    def set(self,value):
        with self.lock: self.value=value
    def get(self):
        with self.lock: return self.value


class VideoReceiver:
    def __init__(self,host,port=9999,codec='h264',raw_path=None,on_event=None,on_frame=None):
        if int(port)!=9999:
            raise ValueError('This iteration accepts only the original video port 9999.')
        self.host=host; self.port=int(port); self.codec=codec; self.raw_path=raw_path
        self.on_event=on_event or (lambda *args:None)
        self.on_frame=on_frame or (lambda frame:None)
        self.frames=LatestValue(); self.stop_event=threading.Event()
        self.sock=None; self.thread=None; self.count=0; self.bytes=0; self.error=None
        self.state='IDLE'; self.started_at=None; self.last_frame_at=None

    def start(self):
        if self.thread and self.thread.is_alive(): raise RuntimeError('Video receiver already active')
        self.thread=threading.Thread(target=self._run,name='msdk-video',daemon=True); self.thread.start()

    def _run(self):
        recording=None; self.started_at=time.monotonic(); self.state='CONNECTING'
        try:
            self.sock=socket.create_connection((self.host,self.port),timeout=4)
            self.sock.settimeout(.5)
            codec=av.CodecContext.create('hevc' if self.codec=='h265' else 'h264','r')
            # Slice threading avoids frame-thread buffering latency.
            codec.thread_count=2; codec.thread_type='SLICE'
            if self.raw_path: recording=open(self.raw_path,'wb')
            self.state='WAITING_VIDEO'; self.on_event('connected',{'host':self.host,'port':self.port})
            connected=time.monotonic(); last_bytes=connected
            while not self.stop_event.is_set():
                try: data=self.sock.recv(262144)
                except socket.timeout:
                    if time.monotonic()-last_bytes>8: raise TimeoutError('No video bytes for 8 seconds. Check the phone video server.')
                    if time.monotonic()-(self.last_frame_at or connected)>8: raise TimeoutError('No decoded frame for 8 seconds. Check H264/H265 selection.')
                    continue
                if not data: raise ConnectionError('Phone closed the video connection.')
                last_bytes=time.monotonic(); self.bytes+=len(data)
                if recording: recording.write(data)
                for packet in codec.parse(data):
                    try: frames=codec.decode(packet)
                    except av.FFmpegError as exc:
                        self.on_event('decode_warning',{'error':str(exc)}); continue
                    for decoded in frames:
                        self.count+=1; self.last_frame_at=time.monotonic()
                        frame=Frame(self.count,self.last_frame_at,decoded.to_ndarray(format='bgr24'))
                        self.frames.set(frame)
                        self.on_frame(frame)
                        self.state='STREAMING'
                if time.monotonic()-(self.last_frame_at or connected)>8:
                    raise TimeoutError('Bytes received but no decoded frames. Check codec selection.')
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error=str(exc); self.state='ERROR'; self.on_event('video_error',{'error':str(exc)})
        finally:
            if recording: recording.close()
            if self.sock:
                try: self.sock.close()
                except OSError: pass
            if self.state!='ERROR': self.state='DISCONNECTED'

    def stop(self):
        self.stop_event.set()
        if self.sock:
            try: self.sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            try: self.sock.close()
            except OSError: pass
        if self.thread: self.thread.join(timeout=5)
