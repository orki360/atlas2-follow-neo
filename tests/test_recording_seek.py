"""Real encoded files: periodic independent seek points and unchanged VFR timing."""
import hashlib
import json
import struct
import tempfile
import time
import unittest
from pathlib import Path

import av
import numpy as np

from follow_neo.recording import VideoRecorder, probe_nvenc
from follow_neo.video import Frame


def atom_order(path):
    atoms=[]
    with path.open('rb') as file:
        end=path.stat().st_size
        while file.tell()<end:
            offset=file.tell()
            size,kind=struct.unpack('>I4s',file.read(8))
            if size==1:size=struct.unpack('>Q',file.read(8))[0]
            if size==0:size=end-offset
            if size<8:raise AssertionError('Invalid MP4 atom')
            atoms.append(kind);file.seek(offset+size)
    return atoms


def nal_types(packet,length_size):
    data=bytes(packet);offset=0;types=[]
    while offset<len(data):
        length=int.from_bytes(data[offset:offset+length_size],'big');offset+=length_size
        if length<=0 or offset+length>len(data):raise AssertionError('Invalid AVC packet')
        types.append(data[offset]&31);offset+=length
    return types


class RecordingSeekTests(unittest.TestCase):
    def exercise(self,profile,expected_encoder):
        # 50 FPS -> 10 FPS -> a source gap -> 30 FPS. Never resample timestamps.
        offsets=[i/50 for i in range(51)]+[1.1+i/10 for i in range(15)]+[3.7+i/30 for i in range(31)]
        expected_pts=[round(t*90000) for t in offsets]
        with tempfile.TemporaryDirectory() as folder:
            recorder=VideoRecorder(folder,'Both videos',profile=profile)
            try:
                for index,t in enumerate(offsets):
                    source=np.full((180,320,3),35,np.uint8)
                    x=(index*5)%280;source[40:90,x:x+40]=(40,180,240)
                    recorder.submit(Frame(index+1,10+t,source),None)
                    deadline=time.monotonic()+10
                    while recorder.written<index+1 and recorder.error is None and time.monotonic()<deadline:
                        time.sleep(.002)
                    self.assertIsNone(recorder.error)
                    self.assertEqual(recorder.written,index+1)
            finally:recorder.wait()
            self.assertIsNone(recorder.error)
            self.assertEqual(recorder.encoder,expected_encoder)
            self.assertEqual(recorder.dropped,0)
            rows=[json.loads(line) for line in (recorder.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([row['pts'] for row in rows],expected_pts)
            requested={row['pts'] for row in rows if row['keyframe_requested']}
            self.assertIn(round(3.7*90000),requested)  # first frame after the gap
            self.assertGreater(len(requested),6)
            for kind in ('clean','prediction'):
                path=recorder.directory/f'{kind}.mp4'
                atoms=atom_order(path)
                self.assertLess(atoms.index(b'moov'),atoms.index(b'mdat'))
                with av.open(str(path)) as container:
                    stream=container.streams.video[0]
                    self.assertEqual(stream.codec_context.name,'h264')
                    length_size=(stream.codec_context.extradata[4]&3)+1
                    packets=[p for p in container.demux(stream) if p.pts is not None]
                    key_pts={p.pts for p in packets if p.is_keyframe}
                    self.assertTrue(requested<=key_pts)
                    last_key=0
                    for packet in packets:
                        if packet.is_keyframe:
                            self.assertIn(5,nal_types(packet,length_size))  # genuine IDR
                            last_key=packet.pts
                        self.assertLess(packet.pts-last_key,45000)
                baseline={}
                with av.open(str(path)) as container:
                    for frame in container.decode(video=0):
                        self.assertEqual((frame.width,frame.height),(320,180))
                        baseline[frame.pts]=hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).digest()
                self.assertEqual(list(baseline),expected_pts)
                with av.open(str(path)) as container:
                    stream=container.streams.video[0]
                    # Reuse the decoder while seeking backwards and forwards.
                    for target in (4.4,1.8,.65,4.1,0.):
                        target_pts=round(target/float(stream.time_base))
                        container.seek(target_pts,stream=stream,backward=True,any_frame=False)
                        decoded=container.decode(stream)
                        first=next(decoded)
                        self.assertTrue(first.key_frame)
                        self.assertLessEqual(first.pts,target_pts)
                        self.assertLessEqual(target_pts-first.pts,45000)
                        found=first
                        while found.pts<target_pts:found=next(decoded)
                        digest=hashlib.sha256(found.to_ndarray(format='rgb24').tobytes()).digest()
                        self.assertEqual(digest,baseline[found.pts])
            report=json.loads((recorder.directory/'recording.json').read_text())
            self.assertEqual(report['state'],'SAVED')
            self.assertEqual(report['keyframe_interval_seconds'],.5)

    def test_software_files_support_independent_seeks_and_preserve_timing(self):
        self.exercise('720p','libx264')

    def test_nvenc_files_support_independent_seeks_and_preserve_timing(self):
        available,reason=probe_nvenc(320,180,2)
        if not available:self.skipTest('NVENC unavailable: '+str(reason))
        self.exercise('1080p','h264_nvenc')


if __name__=='__main__':unittest.main()
