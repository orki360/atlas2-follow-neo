"""Bounded, asynchronous full-resolution MP4 recording with source timestamps.

Both outputs receive identical source frames and PTS. If the bounded queue is
full, both skip the same frame and the skip is reported. No artificial 50 FPS.
"""
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import json
import queue
import threading
import time
import uuid
import av
from .overlay import render_bundle


class VideoRecorder:
    def __init__(self, directory, mode):
        if mode not in ('Clean video','Prediction video','Both videos'):
            raise ValueError('Invalid recording mode')
        self.mode=mode; self.directory=Path(directory)/f'recording_{datetime.now():%H%M%S}_{uuid.uuid4().hex[:6]}'
        self.directory.mkdir(parents=True,exist_ok=False)
        self.queue=queue.Queue(maxsize=8); self.stop_event=threading.Event()
        self.lock=threading.Lock(); self.state='STARTING'; self.error=None
        self.submitted=0; self.written=0; self.dropped=0; self.first_time=None; self.last_time=None
        self.started=time.monotonic(); self.size=None; self._closed=False
        self.thread=threading.Thread(target=self._run,name='video-recorder',daemon=True)
        self.thread.start()

    def submit(self, frame, decision):
        with self.lock:
            if self.stop_event.is_set() or self.error: return
            try: self.queue.put_nowait((frame,decision)); self.submitted+=1
            except queue.Full: self.dropped+=1

    def status(self):
        with self.lock:
            return {'mode':self.mode,'state':self.state,'error':self.error,
                    'written':self.written,'dropped':self.dropped,'path':str(self.directory),
                    'elapsed_seconds':max(0,(self.last_time or self.started)-(self.first_time or self.started))}

    def stop(self):
        with self.lock: self.stop_event.set()

    def wait(self):
        self.stop(); self.thread.join()

    def _run(self):
        outputs={}; metadata=None; last_pts=-1
        try:
            metadata=(self.directory/'frames.jsonl').open('w',encoding='utf-8')
            while not self.stop_event.is_set() or not self.queue.empty():
                try: frame,decision=self.queue.get(timeout=.1)
                except queue.Empty: continue
                h,w=frame.image.shape[:2]
                if self.size is None:
                    self.size=(w,h); self.first_time=frame.decoded_at
                    for kind in ('clean','prediction'):
                        if kind=='clean' and self.mode=='Prediction video': continue
                        if kind=='prediction' and self.mode=='Clean video': continue
                        container=av.open(str(self.directory/f'{kind}.mp4'),'w')
                        try:
                            stream=container.add_stream('libx264',rate=30)
                            stream.width=w; stream.height=h; stream.pix_fmt='yuv420p' if w%2==0 and h%2==0 else 'yuv444p'
                            stream.time_base=Fraction(1,90000); stream.codec_context.time_base=Fraction(1,90000)
                            stream.options={'preset':'ultrafast','crf':'18','tune':'zerolatency'}
                            stream.codec_context.thread_count=2
                            outputs[kind]=(container,stream)
                        except Exception:
                            container.close(); raise
                    self.state='RECORDING'
                if self.size!=(w,h): raise RuntimeError('Video resolution changed; start a new recording.')
                pts=max(last_pts+1,round((frame.decoded_at-self.first_time)*90000));last_pts=pts
                row={'frame_id':frame.frame_id,'decoded_at':frame.decoded_at,'pts':pts,'time_base':'1/90000'}
                for kind,(container,stream) in outputs.items():
                    source=frame.image
                    if kind=='prediction':
                        source,_,overlay=render_bundle(frame,None,decision,'Live prediction',now=frame.decoded_at)
                        row['overlay']=overlay
                    picture=av.VideoFrame.from_ndarray(source,format='bgr24')
                    picture.pts=pts; picture.time_base=Fraction(1,90000)
                    for packet in stream.encode(picture): container.mux(packet)
                metadata.write(json.dumps(row,allow_nan=False)+'\n')
                self.written+=1;self.last_time=frame.decoded_at
            self.state='SAVING'
        except Exception as exc:
            self.error=str(exc); self.state='ERROR'
        finally:
            for container,stream in outputs.values():
                try:
                    for packet in stream.encode(): container.mux(packet)
                    container.close()
                except Exception as exc: self.error=self.error or str(exc)
            if metadata: metadata.close()
            self.state='ERROR' if self.error else 'SAVED'
            report={**self.status(),'resolution':self.size,'codec':'H264 / libx264 / CRF 18',
                    'timing':'local decoder timestamps, not camera exposure',
                    'source_frames_submitted':self.submitted}
            try: (self.directory/'recording.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            except OSError as exc: self.error=str(exc);self.state='ERROR'
