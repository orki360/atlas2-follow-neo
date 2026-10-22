"""Receive-only MSDKRemote elementary H264/H265 TCP video (default 9999).

No control/query connection and no application bytes are ever sent to the phone.
This mirrors VideoServer.java: a raw elementary stream, no length prefix.
"""
from dataclasses import dataclass
import socket
import threading
import time
import av
from .cadence import LatestFrameSampler
from .video_diagnostics import VideoDiagnostics


@dataclass(frozen=True)
class Frame:
    frame_id: int
    decoded_at: float
    image: object
    stream_epoch: int = 0


class LatestValue:
    def __init__(self):
        self.lock=threading.Lock();self.condition=threading.Condition(self.lock)
        self.value=None;self.version=0
    def set(self,value):
        with self.condition:
            self.value=value;self.version+=1;self.condition.notify_all()
    def get(self):
        with self.lock: return self.value
    def wait_next(self,version,timeout):
        with self.condition:
            self.condition.wait_for(lambda:self.version!=version,timeout)
            return self.value,self.version


class VideoReceiver:
    def __init__(self,host,port=9999,codec='h264',raw_path=None,on_event=None,on_frame=None,processing_fps=30.):
        self.raw_recording_enabled=True
        if int(port)!=9999:
            raise ValueError('This iteration accepts only the original video port 9999.')
        self.host=host; self.port=int(port); self.codec=codec; self.raw_path=raw_path
        self.on_event=on_event or (lambda *args:None)
        self.on_frame=on_frame or (lambda frame:None)
        self.frames=LatestValue(); self.stop_event=threading.Event()
        self.sock=None; self.thread=None; self.count=0; self.bytes=0; self.error=None
        self.state='IDLE'; self.started_at=None; self.last_frame_at=None
        self.stream_epoch=0;self.reconnect_count=0;self.consecutive_failures=0
        self.processing_fps=processing_fps;self.processed_count=0;self.sampled_out=0
        self.sampler=LatestFrameSampler()
        self.sample_condition=threading.Condition()
        self.publication_lock=threading.Lock()
        self.publisher_thread=None
        self.diagnostics=VideoDiagnostics();self.monitor_thread=None
        self.timing=dict(last_decode_gap_ms=None,max_decode_gap_ms=0.,
            decode_gaps_over_250ms=0,decode_gaps_over_500ms=0,
            last_receive_at=None,last_receive_wait_ms=None,last_decode_batch_ms=None,last_publish_ms=None)

    def timing_snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        timing=dict(self.timing);received=timing['last_receive_at']
        with self.sample_condition:sampling=self.sampler.snapshot()
        timing.update(sampling=sampling,
            decoder_frame_age_ms=max(0.,now-self.last_frame_at)*1000 if self.last_frame_at is not None else None,
            bytes_age_ms=max(0.,now-received)*1000 if received is not None else None,
            stream_epoch=self.stream_epoch,reconnect_count=self.reconnect_count,
            timestamp_origin='local_decoder_monotonic_not_camera_exposure')
        return timing

    def _publish_decoded(self,decoded,stamp):
        previous=self.last_frame_at
        if previous is not None:
            gap=max(0.,stamp-previous)*1000
            timing=dict(self.timing);timing.update(last_decode_gap_ms=gap,
                max_decode_gap_ms=max(timing['max_decode_gap_ms'],gap),
                decode_gaps_over_250ms=timing['decode_gaps_over_250ms']+int(gap>250),
                decode_gaps_over_500ms=timing['decode_gaps_over_500ms']+int(gap>500))
            self.timing=timing
            if gap>250:
                self.on_event('video_decode_gap',dict(previous_decoded_at=previous,decoded_at=stamp,
                    gap_ms=gap,decoded_frame_id=self.count+1,timing=self.timing_snapshot(stamp),
                    pipeline=self.diagnostics.snapshot()))
        self.count+=1;self.last_frame_at=stamp;self.state='STREAMING'
        self.diagnostics.decoded(getattr(decoded,'key_frame',False),stamp)
        if self.error:
            self.on_event('video_recovered',dict(stream_epoch=self.stream_epoch,
                reconnect_count=self.reconnect_count,previous_error=self.error))
        self.error=None;self.consecutive_failures=0
        # All reference frames are still decoded with the unchanged codec.
        # Only one decoded frame is retained; conversion is outside recv/decode.
        with self.sample_condition:
            self.sampler.offer((decoded,self.count,stamp,self.stream_epoch),stamp)
            self.sampled_out=self.sampler.replaced+self.sampler.expired+self.sampler.invalidated
            self.sample_condition.notify_all()

    def _publish_pending(self,now=None):
        with self.sample_condition:
            # Read after taking the lock: a newly offered frame must not appear
            # to come from the future because this worker was descheduled.
            now=time.monotonic() if now is None else now
            item=self.sampler.take(now,self.processing_fps)
            self.sampled_out=self.sampler.replaced+self.sampler.expired+self.sampler.invalidated
        if item is None:return False
        decoded,frame_id,stamp,epoch=item
        with self.diagnostics.measure('convert_bgr'):
            frame=Frame(frame_id,stamp,decoded.to_ndarray(format='bgr24'),epoch)
        # A reconnect or expensive conversion cannot resurrect an old frame.
        with self.publication_lock:
            age=time.monotonic()-stamp
            if (self.stop_event.is_set() or epoch!=self.stream_epoch or self.state!='STREAMING'
                    or not 0<=age<=self.sampler.MAX_AGE_SECONDS):
                self.diagnostics.count('publication_discarded');return False
            self.processed_count+=1
            with self.diagnostics.measure('publish_callback'):
                self.frames.set(frame);self.on_frame(frame)
            self.timing=dict(self.timing,last_publish_ms=max(0.,time.monotonic()-stamp)*1000)
        return True

    def _publish_loop(self):
        try:
            while not self.stop_event.is_set():
                self._publish_pending()
                with self.sample_condition:
                    if self.stop_event.is_set():break
                    self.sample_condition.wait(self.sampler.wait_seconds(time.monotonic()))
        except Exception as exc:
            self._invalidate_stream(exc)
            self.stop_event.set()
            if self.sock:
                try:self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass

    def start(self):
        if self.thread and self.thread.is_alive(): raise RuntimeError('Video receiver already active')
        self.publisher_thread=threading.Thread(target=self._publish_loop,name='video-publisher',daemon=True)
        self.publisher_thread.start()
        self.thread=threading.Thread(target=self._run,name='msdk-video',daemon=True); self.thread.start()
        self.monitor_thread=threading.Thread(target=self._monitor,name='video-health',daemon=True)
        self.monitor_thread.start()

    def _health(self,reason):
        self.on_event('video_health',dict(reason=reason,state=self.state,error=self.error,
            total_bytes=self.bytes,total_decoded=self.count,total_processed=self.processed_count,
            timing=self.timing_snapshot(),pipeline=self.diagnostics.snapshot(reset=True)))

    def _monitor(self):
        # Separate from recv/decode: it can report a stalled receiver stage.
        while not self.stop_event.wait(1.):
            self._health('periodic')
            if not self.thread.is_alive():break

    def _run(self):
        recording=None; self.started_at=time.monotonic(); self.state='CONNECTING'
        try:
            if self.raw_path and self.raw_recording_enabled: recording=open(self.raw_path,'wb')
            while not self.stop_event.is_set():
                try:
                    self._receive_stream(recording)
                except OSError as exc:
                    if self.stop_event.is_set():break
                    self.consecutive_failures+=1
                    delay=min(4.,.5*2**min(self.consecutive_failures-1,3))
                    self._invalidate_stream(exc,retry_seconds=delay)
                else:
                    break
                finally:
                    if self.sock:
                        try:self.sock.close()
                        except OSError:pass
                    self.sock=None
                # Receive-only reconnect, with bounded backoff and a new codec.
                # MSDKRemote starts a new client at its next keyframe.
                if self.stop_event.wait(delay):break
                self.reconnect_count+=1
        except Exception as exc:
            if not self.stop_event.is_set():
                self._invalidate_stream(exc)
        finally:
            if recording: recording.close()
            if self.sock:
                try: self.sock.close()
                except OSError: pass
            if self.state!='ERROR': self.state='DISCONNECTED'
            self.stop_event.set()
            with self.sample_condition:
                self.sampler.clear();self.sample_condition.notify_all()

    def _invalidate_stream(self,exc,retry_seconds=None):
        with self.publication_lock:
            self.error=str(exc);self.state='RECONNECTING' if retry_seconds is not None else 'ERROR'
            self.stream_epoch+=1
            self.frames.set(None)
            with self.sample_condition:
                self.sampler.clear();self.sample_condition.notify_all()
        self.on_event('video_error',dict(error=self.error,retrying=retry_seconds is not None,
            retry_seconds=retry_seconds,stream_epoch=self.stream_epoch,timing=self.timing_snapshot()))
        self._health('stream_error')

    def _receive_stream(self,recording):
        with self.diagnostics.measure('connect'):
            self.sock=socket.create_connection((self.host,self.port),timeout=4)
        self.sock.settimeout(.5)
        if self.stop_event.is_set():return
        codec=av.CodecContext.create('hevc' if self.codec=='h265' else 'h264','r')
        # Slice threading avoids frame-thread buffering latency.
        codec.thread_count=2;codec.thread_type='SLICE'
        self.state='WAITING_VIDEO'
        self.on_event('connected',dict(host=self.host,port=self.port,stream_epoch=self.stream_epoch,
            reconnect_count=self.reconnect_count))
        connected=time.monotonic();last_bytes=connected;last_decoded=connected
        self.diagnostics.connected(connected)
        while not self.stop_event.is_set():
            # Close on the receive worker, never wait for disk I/O in the UI.
            # Once stopped, raw recording cannot resume in this connection.
            if recording and not self.raw_recording_enabled:
                recording.close();recording=None
            receive_started=time.monotonic()
            try:
                with self.diagnostics.measure('receive'):data=self.sock.recv(262144)
            except socket.timeout:
                self.diagnostics.count('receive_timeouts')
                if time.monotonic()-last_bytes>8:raise TimeoutError('No video bytes for 8 seconds. Check the phone video server.')
                if time.monotonic()-last_decoded>8:raise TimeoutError('No decoded frame for 8 seconds. Check H264/H265 selection.')
                continue
            if not data:raise ConnectionError('Phone closed the video connection.')
            last_bytes=time.monotonic();self.bytes+=len(data)
            self.diagnostics.received(len(data),last_bytes)
            self.timing=dict(self.timing,last_receive_at=last_bytes,
                last_receive_wait_ms=max(0.,last_bytes-receive_started)*1000)
            if recording and self.raw_recording_enabled:
                with self.diagnostics.measure('raw_record_write'):recording.write(data)
            with self.diagnostics.measure('parse'):packets=codec.parse(data)
            self.diagnostics.count('packets',len(packets))
            for packet in packets:
                try:
                    with self.diagnostics.measure('decode'):frames=codec.decode(packet)
                except av.FFmpegError as exc:
                    self.diagnostics.count('decode_errors')
                    self.on_event('decode_warning',{'error':str(exc)});continue
                for decoded in frames:
                    last_decoded=time.monotonic()
                    self._publish_decoded(decoded,last_decoded)
            self.timing=dict(self.timing,last_decode_batch_ms=max(0.,time.monotonic()-last_bytes)*1000)
            if time.monotonic()-last_decoded>8:
                raise TimeoutError('Bytes received but no decoded frames. Check codec selection.')

    def stop(self):
        self.stop_event.set()
        with self.sample_condition:
            self.sampler.clear();self.sample_condition.notify_all()
        sock=self.sock
        if sock:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            try: sock.close()
            except OSError: pass
        if self.thread: self.thread.join(timeout=5)
        if self.publisher_thread:self.publisher_thread.join(timeout=2)
        if self.monitor_thread:self.monitor_thread.join(timeout=2)
