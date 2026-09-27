"""Bounded asynchronous recording, paired timestamps and measured encoder load.

Auto probes simultaneous NVENC encoders; if unavailable it explicitly selects
720p software encoding. 1080p preserves native resolution (no silent downscale).
Capture, inference and control never wait for this worker's encoding queue.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import json,queue,threading,time,uuid
import av
import cv2
import numpy as np
from .overlay import render_bundle

TIME_BASE=Fraction(1,90000)
_PROBES={}
_PROBE_LOCK=threading.Lock()


def encoder_options(name):
    if name=='h264_nvenc':return {'preset':'p1','tune':'ll','rc':'constqp','qp':'23','bf':'0'}
    return {'preset':'ultrafast','crf':'20','tune':'zerolatency','bf':'0'}


def probe_nvenc(width,height,count=2):
    key=(width,height,count)
    with _PROBE_LOCK:
        if key in _PROBES:return _PROBES[key]
        contexts=[]
        try:
            # Codec registration alone is not evidence of a working device.
            for _ in range(count):
                c=av.CodecContext.create('h264_nvenc','w');contexts.append(c)
                c.width=width;c.height=height;c.pix_fmt='yuv420p';c.time_base=TIME_BASE
                c.options=encoder_options('h264_nvenc');c.open()
            for c in contexts:
                f=av.VideoFrame.from_ndarray(np.zeros((height*3//2,width),np.uint8),format='yuv420p')
                f.pts=0;f.time_base=TIME_BASE
                packets=c.encode(f)+c.encode(None)
                if not packets:raise RuntimeError('NVENC probe produced no packet')
            result=(True,None)
        except Exception as exc:result=(False,str(exc)[:240])
        finally:contexts.clear()
        _PROBES[key]=result
        return result


def recording_size(width,height,profile,hardware):
    if profile not in ('Auto','1080p','720p'):raise ValueError('Unknown recording profile')
    if profile=='720p' or profile=='Auto' and not hardware:
        scale=min(1.,1280/width,720/height)
        return (max(2,int(width*scale)//2*2),max(2,int(height*scale)//2*2))
    return width,height


class VideoRecorder:
    def __init__(self,directory,mode,profile='1080p',on_event=None):
        if mode not in ('Clean video','Prediction video','Both videos'):raise ValueError('Invalid recording mode')
        if profile not in ('Auto','1080p','720p'):raise ValueError('Unknown recording profile')
        self.mode=mode;self.profile=profile;self.on_event=on_event or (lambda *a:None)
        self.directory=Path(directory)/f'recording_{datetime.now():%H%M%S}_{uuid.uuid4().hex[:6]}'
        self.directory.mkdir(parents=True,exist_ok=False)
        self.queue=queue.Queue(maxsize=8);self.stop_event=threading.Event();self.lock=threading.Lock()
        self.state='STARTING';self.error=None;self.submitted=0;self.written=0;self.dropped=0
        self.first_time=None;self.last_time=None;self.started=time.monotonic();self.size=None;self.source_size=None
        self.encoder=None;self.fallback_reason=None;self.encode_times=deque(maxlen=120);self.queue_times=deque(maxlen=120)
        self.thread=threading.Thread(target=self._run,name='video-recorder',daemon=True);self.thread.start()

    def submit(self,frame,decision):
        with self.lock:
            if self.stop_event.is_set() or self.error:return
            try:self.queue.put_nowait((frame,decision,time.monotonic()));self.submitted+=1
            except queue.Full:self.dropped+=1

    def status(self):
        with self.lock:
            elapsed=max(0.,(self.last_time or self.started)-(self.first_time or self.started))
            return dict(mode=self.mode,state=self.state,error=self.error,written=self.written,dropped=self.dropped,
                path=str(self.directory),elapsed_seconds=elapsed,profile=self.profile,encoder=self.encoder,
                resolution=self.size,source_resolution=self.source_size,fallback_reason=self.fallback_reason,
                recorded_fps=(self.written-1)/elapsed if elapsed>0 and self.written else 0.,
                encode_ms_mean=sum(self.encode_times)/len(self.encode_times) if self.encode_times else 0.,
                queue_age_ms_max=max(self.queue_times,default=0.),queue_depth=self.queue.qsize())

    def stop(self):
        with self.lock:self.stop_event.set()

    def wait(self):self.stop();self.thread.join()

    def _open_outputs(self,kinds,width,height):
        outputs={}
        try:
            for kind in kinds:
                container=av.open(str(self.directory/f'{kind}.mp4'),'w')
                stream=container.add_stream(self.encoder,rate=30);outputs[kind]=(container,stream)
                stream.width=width;stream.height=height
                stream.pix_fmt='yuv420p' if width%2==0 and height%2==0 else 'yuv444p'
                stream.time_base=TIME_BASE;stream.codec_context.time_base=TIME_BASE
                stream.options=encoder_options(self.encoder);stream.codec_context.thread_count=2
                stream.codec_context.open()
            return outputs
        except Exception:
            for container,_ in outputs.values():container.close()
            raise

    @staticmethod
    def _encode(output,source,pts):
        container,stream=output
        # Fast explicit conversion avoids two costly implicit swscale paths.
        if source.shape[0]%2==0 and source.shape[1]%2==0:
            picture=av.VideoFrame.from_ndarray(cv2.cvtColor(source,cv2.COLOR_BGR2YUV_I420),format='yuv420p')
        else:picture=av.VideoFrame.from_ndarray(source,format='bgr24')
        picture.pts=pts;picture.time_base=TIME_BASE
        for packet in stream.encode(picture):container.mux(packet)

    def _run(self):
        outputs={};metadata=None;last_pts=-1;last_event=0.
        pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='video-encode')
        try:
            metadata=(self.directory/'frames.jsonl').open('w',encoding='utf-8')
            while not self.stop_event.is_set() or not self.queue.empty():
                try:frame,decision,enqueued=self.queue.get(timeout=.1)
                except queue.Empty:continue
                h,w=frame.image.shape[:2]
                if self.size is None:
                    self.source_size=(w,h);self.first_time=frame.decoded_at
                    kinds=['clean','prediction'] if self.mode=='Both videos' else ['clean' if self.mode=='Clean video' else 'prediction']
                    hardware,self.fallback_reason=probe_nvenc(w,h,len(kinds)) if self.profile!='720p' else (False,'720p software profile selected')
                    self.encoder='h264_nvenc' if hardware else 'libx264'
                    self.size=recording_size(w,h,self.profile,hardware)
                    try:outputs=self._open_outputs(kinds,*self.size)
                    except Exception as exc:
                        if not hardware:raise
                        self.fallback_reason='NVENC output initialization failed: '+str(exc)[:180]
                        self.encoder='libx264';self.size=recording_size(w,h,self.profile,False)
                        outputs=self._open_outputs(kinds,*self.size)
                    self.state='RECORDING';self.on_event('recording_encoder',self.status())
                if self.source_size!=(w,h):raise RuntimeError('Video resolution changed; start a new recording.')
                pts=max(last_pts+1,round((frame.decoded_at-self.first_time)*90000));last_pts=pts
                row=dict(frame_id=frame.frame_id,decoded_at=frame.decoded_at,pts=pts,time_base='1/90000',resolution=self.size)
                started=time.monotonic();sources={}
                if 'clean' in outputs:
                    sources['clean']=frame.image if self.size==(w,h) else cv2.resize(frame.image,self.size,interpolation=cv2.INTER_AREA)
                if 'prediction' in outputs:
                    sources['prediction'],_,overlay=render_bundle(frame,None,decision,'Live prediction',now=frame.decoded_at,output_size=self.size)
                    row['overlay']=overlay
                futures=[pool.submit(self._encode,output,sources[kind],pts) for kind,output in outputs.items()]
                for future in futures:future.result()
                ended=time.monotonic()
                row.update(encode_ms=(ended-started)*1000,queue_age_ms=(started-enqueued)*1000)
                metadata.write(json.dumps(row,allow_nan=False)+'\n')
                with self.lock:
                    self.written+=1;self.last_time=frame.decoded_at
                    self.encode_times.append(row['encode_ms']);self.queue_times.append(row['queue_age_ms'])
                if ended-last_event>=1:
                    self.on_event('recording_metrics',self.status());last_event=ended
            self.state='SAVING'
        except Exception as exc:self.error=str(exc);self.state='ERROR'
        finally:
            pool.shutdown(wait=True)
            for container,stream in outputs.values():
                try:
                    for packet in stream.encode():container.mux(packet)
                    container.close()
                except Exception as exc:self.error=self.error or str(exc)
            if metadata:metadata.close()
            self.state='ERROR' if self.error else 'SAVED'
            report={**self.status(),'codec':self.encoder,'timing':'source decoder timestamps; dropped frames leave time gaps',
                    'source_frames_submitted':self.submitted}
            try:(self.directory/'recording.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            except OSError as exc:self.error=str(exc);self.state='ERROR'
            self.on_event('recorder_finalized',self.status())
