"""Horizontal search, explicit yaw override, frame cadence and close-latch regressions."""
import sys,threading,time,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.types import Box,Detection,Settings,Intent,settings_from_config
from follow_neo.controller import FollowController
from follow_neo.spacing import VisualSpacingController
from follow_neo.dance import dance_command,ZERO
from follow_neo.edge_search import yaw_override
from follow_neo.video import LatestValue,VideoReceiver,Frame
from follow_neo.session import FollowSession
from follow_neo.cadence import FrameRateGate
from follow_neo.manual import ManualControl
from test_manual import Server,wait_for


class EdgeSearchTests(unittest.TestCase):
    def ready(self,side=-1,settings=None):
        c=FollowController();s=settings or Settings(confidence=.3,stop_width=.34)
        for i,cx in enumerate((500,420,330,250,190,140,110,85)):
            if side>0:cx=1280-cx
            stamp=10+i*.05
            c.observe([Detection(Box(cx-35,320,cx+35,380),.85)],stamp,stamp+.02,i+1,(1280,720),s)
            d=c.tick(stamp+.02,1280,720,stamp+.02,s)
        self.assertEqual(d['state'],'TRACK');return c,s

    def gap(self,c,s,now=10.42,fid=100,video_time=None):
        c.observe([],now-.02,now,fid,(1280,720),s)
        return c.tick(now,1280,720,now if video_time is None else video_time,s)

    def test_left_and_right_exits_use_yaw_only_at_full_scale(self):
        for side in (-1,1):
            c,s=self.ready(side);d=self.gap(c,s)
            self.assertTrue(d['edge_search']['active'])
            self.assertEqual(dance_command(d,10.42,9)[0],(float(side),0.,0.,0.))

    def test_regular_tracking_still_uses_ordinary_gate(self):
        c,s=self.ready();d=c.tick(10.38,1280,720,10.38,s)
        self.assertFalse(d['edge_search']['active'])
        self.assertFalse(dance_command(d,10.38,9)[1].startswith('Edge search'))

    def test_deadline_does_not_move_and_cannot_restart_without_detections(self):
        c,s=self.ready();first=self.gap(c,s);end=first['edge_search']['until']
        for i in range(1,15):
            now=10.42+i*.10;d=self.gap(c,s,now,100+i)
            if now<end:self.assertEqual(d['edge_search']['until'],end)
            else:
                self.assertFalse(d['edge_search']['active'])
                self.assertEqual(dance_command(d,now,9)[0],ZERO)

    def test_reacquisition_cancels_full_yaw_on_first_accepted_measurement(self):
        c,s=self.ready();self.gap(c,s)
        c.observe([Detection(Box(80,320,150,380),.9)],10.46,10.48,101,(1280,720),s)
        d=c.tick(10.48,1280,720,10.48,s)
        self.assertTrue(d['accepted']);self.assertFalse(d['edge_search']['active'])
        self.assertFalse(dance_command(d,10.48,9)[1].startswith('Edge search'))

    def test_competing_detection_cancels_search(self):
        c,s=self.ready();self.gap(c,s)
        c.observe([Detection(Box(1000,20,1030,50),.95)],10.46,10.48,101,(1280,720),s)
        d=c.tick(10.48,1280,720,10.48,s)
        self.assertFalse(d['edge_search']['active']);self.assertEqual(yaw_override(d,10.48),0)

    def test_center_frame_miss_does_not_start_full_yaw(self):
        c=FollowController();s=Settings()
        for i in range(8):
            t=10+i*.05
            c.observe([Detection(Box(600,310,680,370),.9)],t,t+.02,i+1,(1280,720),s)
            c.tick(t+.02,1280,720,t+.02,s)
        self.assertFalse(self.gap(c,s)['edge_search']['active'])

    def test_disabled_option_stale_video_and_near_target_block_search(self):
        c,s=self.ready(settings=Settings(edge_search_enabled=False));self.assertFalse(self.gap(c,s)['edge_search']['active'])
        c,s=self.ready();self.assertFalse(self.gap(c,s,video_time=9)['edge_search']['active'])
        c,s=self.ready();c.spacing.close=True
        self.assertFalse(self.gap(c,s)['edge_search']['active'])

    def test_vertical_only_loss_does_not_invent_horizontal_search(self):
        c=FollowController();s=Settings()
        for i,cy in enumerate((400,330,250,200,150,100,60,30)):
            t=10+i*.05
            c.observe([Detection(Box(600,cy-20,680,cy+20),.9)],t,t+.02,i+1,(1280,720),s)
            c.tick(t+.02,1280,720,t+.02,s)
        self.assertFalse(self.gap(c,s)['edge_search']['active'])

    def test_search_cannot_cross_send_deadline_or_new_dance_start(self):
        c,s=self.ready();d=self.gap(c,s)
        self.assertEqual(dance_command(d,10.42,10.43)[0],ZERO)
        d['prediction_time']=10.50
        self.assertEqual(dance_command(d,10.50,10.45)[0],ZERO)
        d['prediction_time']=d['edge_search']['until']-.01
        self.assertEqual(dance_command(d,d['edge_search']['until']+.001,9)[0],ZERO)

    def test_malformed_or_stale_search_cannot_override_axis_limits(self):
        c,s=self.ready();d=self.gap(c,s)
        for change in ({'accepted':True},{'stale':True},{'state':'ERROR'},
                       {'spacing_phase':'STOPPED'},{'spacing_close_guard_armed':True}):
            self.assertEqual(yaw_override({**d,**change},10.42),0.)
        for change in ({'direction':2},{'until':float('inf')},{'until':20},{'source_time':9}):
            self.assertEqual(yaw_override({**d,'edge_search':{**d['edge_search'],**change}},10.42),0.)

    def test_actual_wire_overrides_yaw_only_and_returns_to_slider(self):
        c,s=self.ready();template=self.gap(c,s);state={'search':True}
        server=Server();events=[]
        def provider():
            now=time.monotonic()
            if state['search']:
                return {**template,'prediction_time':now,'edge_search':{
                    **template['edge_search'],'started':now-.01,'source_time':now-.05,'until':now+.2}}
            return {'prediction_time':now,'measurement_age_ms':20,'accepted':True,'confirmed':True,
                'state':'TRACK','spacing_phase':'APPROACH','stale':False,
                'intent':dict(yaw=-1.,vertical=.5,roll=.5,forward=.5),
                'intent_basis':dict(yaw=1.,vertical=1.,roll=1.,forward=1.)}
        control=ManualControl('127.0.0.1',server.connect,decision_provider=provider,
                    on_event=lambda event,data:events.append((event,data)))
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.05,vertical=.5,roll=.015,forward=.015))
        control.start();control.update(set());control.enable();wait_for(lambda:control.enabled);control.start_dance()
        wait_for(lambda:'rc -1.0000 0.0000 0.0000 0.0000' in server.commands)
        wait_for(lambda:any(e=='flight_command' and d.get('edge_yaw_override') for e,d in events))
        edge=next(d for e,d in events if e=='flight_command' and d.get('edge_yaw_override'))
        self.assertEqual(edge['axis_limits']['yaw'],.05);self.assertEqual(edge['effective_axis_limits']['yaw'],1.)
        state['search']=False;control.update(set());offset=len(events)
        wait_for(lambda:any(e=='flight_command' and not d.get('edge_yaw_override') for e,d in events[offset:]))
        normal=next(d for e,d in events[offset:] if e=='flight_command' and not d.get('edge_yaw_override'))
        self.assertLessEqual(abs(normal['values'][0]),.05)
        self.assertLessEqual(abs(normal['values'][2]),.015)

    def test_keyboard_override_uses_manual_yaw_cap_even_during_search(self):
        c,s=self.ready();template=self.gap(c,s);server=Server();events=[]
        def provider():
            now=time.monotonic()
            return {**template,'prediction_time':now,'edge_search':{
                **template['edge_search'],'started':now-.01,'source_time':now-.05,'until':now+.2}}
        control=ManualControl('127.0.0.1',server.connect,decision_provider=provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.05,vertical=.5,roll=.015,forward=.015))
        control.start();control.update(set());control.enable();wait_for(lambda:control.enabled);control.start_dance()
        control.update({'d'})
        wait_for(lambda:'rc 0.0500 0.0000 0.0000 0.0000' in server.commands)


class CadenceTests(unittest.TestCase):
    def test_sixty_source_frames_sample_to_thirty_without_cumulative_drift(self):
        for hz in (60.,59.8):
            gate=FrameRateGate()
            count=sum(gate.admit(10+i/hz,30.) for i in range(round(hz*10)))
            self.assertLessEqual(abs(count-300),1)

    def test_quantized_arrival_times_still_maintain_average_target(self):
        gate=FrameRateGate()
        count=sum(gate.admit(10+round(i/60/.015625)*.015625,30.) for i in range(600))
        self.assertLessEqual(abs(count-300),1)

    def test_stalled_input_does_not_publish_a_catchup_burst(self):
        gate=FrameRateGate();self.assertTrue(gate.admit(1,30));self.assertTrue(gate.admit(4,30))
        self.assertFalse(gate.admit(4.001,30));self.assertFalse(gate.admit(4.002,30))

    def test_receiver_skips_conversion_for_unsampled_frames(self):
        decoded=Mock();decoded.to_ndarray.return_value=np.zeros((20,20,3),np.uint8)
        published=[];r=VideoReceiver('127.0.0.1',processing_fps=30,on_frame=published.append)
        for i in range(120):r._publish_decoded(decoded,10+i/60)
        self.assertEqual(r.count,120);self.assertEqual(r.processed_count,60)
        self.assertEqual(r.sampled_out,60);self.assertEqual(decoded.to_ndarray.call_count,60)
        self.assertEqual(len(published),60);self.assertEqual(r.frames.version,60)
        self.assertEqual(published[1].frame_id,3)

    def test_latest_value_wakes_consumer_and_does_not_build_backlog(self):
        latest=LatestValue();result=[];waiting=threading.Event()
        def consume():
            waiting.set();result.append(latest.wait_next(0,1))
        thread=threading.Thread(target=consume);thread.start();self.assertTrue(waiting.wait(1))
        latest.set('frame');thread.join(2)
        self.assertEqual(result,[('frame',1)])
        latest.set('older');latest.set('newest')
        self.assertEqual(latest.wait_next(1,0),('newest',3))
        self.assertEqual(latest.wait_next(3,0),('newest',3))

    def test_config_migration_is_once_and_preserves_tracking_values(self):
        config={'schema_version':3,'settings':{'inference_fps':60,'stop_width':.34,'confidence':.3},
                'axis_limits':{'yaw':.5}}
        s=settings_from_config(config)
        self.assertEqual((s.video_fps,s.inference_fps),(30,30))
        self.assertEqual((s.stop_width,s.confidence),(.34,.3))
        self.assertEqual(config['settings']['inference_fps'],60)
        config['schema_version']=4;config['settings'].update(video_fps=25,inference_fps=20,edge_search_enabled=False)
        s=settings_from_config(config)
        self.assertEqual((s.video_fps,s.inference_fps),(25,20));self.assertFalse(s.edge_search_enabled)

    def test_invalid_rate_or_search_duration_is_rejected(self):
        for change in ({'video_fps':0},{'inference_fps':61},{'edge_search_seconds':3},
                       {'edge_search_seconds':float('nan')},{'edge_search_enabled':'false'}):
            with self.assertRaises(ValueError):replace(Settings(),**change).validate()


class CloseLatchTests(unittest.TestCase):
    def test_repeated_distant_measurements_clear_historical_close_latch(self):
        c=VisualSpacingController();s=Settings(stop_width=.34)
        def step(t,mid,width,rejected=False,clipped=False):
            return c.update(t,True,True,rejected,clipped,False,mid,t,0,width,width,0,0,Intent(forward=1),s)
        step(10,1,.21);self.assertTrue(c.close)
        for i in range(5):step(10.1+i*.05,i+2,.09)
        self.assertFalse(c.close)
        step(10.4,10,.09,clipped=True)
        self.assertEqual(c.phase,'APPROACH')

    def test_predictions_and_rejected_frames_never_clear_close_latch(self):
        for rejected in (False,True):
            c=VisualSpacingController();s=Settings(stop_width=.34)
            c.update(10,True,True,False,False,False,1,10,0,.21,.21,0,0,Intent(forward=1),s)
            for i in range(1,4):
                c.update(10+i*.05,True,True,rejected,False,False,1,10,i*.05,.09,.09,0,0,Intent(forward=1),s)
            self.assertTrue(c.close)

    def test_large_clipped_target_remains_a_terminal_stop(self):
        c=VisualSpacingController();s=Settings(stop_width=.34)
        for i in range(4):
            c.update(10+i*.1,True,True,False,False,False,i+1,10+i*.1,0,.23,.23,0,0,Intent(),s)
        c.update(10.4,True,True,False,True,False,5,10.4,0,.40,.40,0,0,Intent(),s)
        self.assertEqual(c.phase,'STOPPED')


class SessionSchedulingTests(unittest.TestCase):
    def run_pipeline(self,reconfigure):
        """Real worker threads with a controlled detector; never open a socket."""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        session=FollowSession(temp.name,'127.0.0.1')
        entered=threading.Event();release=threading.Event();seen=[];published=[]
        detector=Mock();detector.input.shape=[1,3,640,640];detector.output.shape=[1,5,8400]
        detector.session.get_providers.return_value=['TestProvider']
        detector.class_names=['drone'];detector.compute_info={'label':'test double'}
        def detect(image):
            seen.append(int(image[0,0,0]))
            if len(seen)==1:
                entered.set()
                if not release.wait(3):raise RuntimeError('test detector release timed out')
            return dict(detections=[],inference_ms=1.,preprocess_ms=1.,postprocess_ms=0.)
        detector.detect.side_effect=detect
        original_set=session.result.set
        def publish(value):
            published.append(value['frame'].frame_id);original_set(value)
        session.result.set=publish
        def cleanup():
            release.set();session.stop()
        self.addCleanup(cleanup)
        def frame(fid):
            session.receiver.frames.set(Frame(fid,time.monotonic(),np.full((20,20,3),fid,np.uint8)))
        with patch('follow_neo.session.OnnxDroneDetector',return_value=detector):
            session.receiver.state='STREAMING'
            session.inference_thread.start();session.control_thread.start()
            frame(1);self.assertTrue(entered.wait(2))
            if reconfigure:
                session.configure(replace(session.get_settings(),confidence=.3))
                wait_for(lambda:not session.reset_event.is_set())
            frame(2);frame(3);release.set()
            wait_for(lambda:session.analysis.get() is not None and
                     session.analysis.get()['result']['frame'].frame_id==3 and
                     session.inference_count==2)
        self.assertIsNone(session.model_error)
        self.assertEqual(seen,[1,3]);self.assertEqual(session.inference_skips,1)
        self.assertEqual(session.inference_count,2)
        self.assertEqual(session.analysis.get()['result']['generation'],int(reconfigure))
        if reconfigure:self.assertEqual(published,[3])
        else:self.assertEqual(published,[1,3])

    def test_workers_consume_latest_frame_without_building_a_queue(self):
        self.run_pipeline(False)

    def test_inflight_result_from_old_settings_is_never_published(self):
        self.run_pipeline(True)


if __name__=='__main__':unittest.main()
