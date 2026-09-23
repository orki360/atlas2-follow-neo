"""Update 9: continuous Dance, measured search arcs, prediction and final wire caps.

All transport tests use local socket pairs; no aircraft or phone is contacted.
"""
import json,math,socket,sys,tempfile,threading,time,unittest
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_neo.types import Box,Detection,Settings,Intent,Kinematics,settings_from_config
from follow_neo.controller import FollowController
from follow_neo.tracker import BBoxTracker
from follow_neo.spacing import VisualSpacingController
from follow_neo.edge_search import EdgeYawSearch,yaw_override
from follow_neo.dance import dance_command,ZERO,DanceSmoother
from follow_neo.manual import ManualControl,DEFAULT_AXIS_LIMITS,movement
from follow_neo.telemetry import HeadingReceiver,parse_heading,fresh_heading
from follow_neo.gui import FollowLabWindow,normalized_event_key,safe_focus_widget


def heading(t,yaw=0):return dict(time=t,yaw_deg=yaw,source='AircraftAttitude')

def wait_for(predicate,seconds=2.):
    end=time.monotonic()+seconds
    while not predicate():
        if time.monotonic()>end:raise AssertionError('Timed out waiting for local fixture')
        time.sleep(.005)


class Fixture:
    def __init__(self,width=.07,move=0.,side=1,settings=None):
        self.c=FollowController();self.s=settings or Settings(stop_width=.24)
        self.id=0;self.now=10.;self.side=side
        for i in range(15):self.step(width=width,cx=.50+side*move*i)

    def step(self,width=.07,cx=.55,missing=False,stale=False,yaw=None,dt=1/30):
        self.now+=dt;self.id+=1;t=self.now
        ds=[] if missing else [Detection(Box((cx-width/2)*1280,320,(cx+width/2)*1280,380),.9)]
        self.c.observe(ds,t,t+.015,self.id,(1280,720),self.s)
        self.d=self.c.tick(t+.015,1280,720,t-3 if stale else t,self.s,
                           None if yaw is None else heading(t,yaw))
        return self.d


class ContinuousTests(unittest.TestCase):
    def test_short_empty_result_keeps_forward_and_real_timestamp(self):
        f=Fixture();self.assertEqual(f.d['state'],'TRACK')
        mt=f.d['measurement_time'];mid=f.d['measurement_id']
        d=f.step(missing=True)
        values,status=dance_command(d,f.now+.015,9)
        self.assertEqual(d['state'],'COAST');self.assertGreater(values[3],0)
        self.assertEqual((d['measurement_time'],d['measurement_id']),(mt,mid))
        self.assertTrue(status.startswith('Kalman recovery'))
        self.assertFalse(d['accepted'])

    def test_one_frame_gap_returns_directly_to_track(self):
        f=Fixture();f.step(missing=True);d=f.step()
        self.assertEqual(d['state'],'TRACK');self.assertGreater(d['intent']['forward'],0)

    def test_long_loss_never_ends_continuous_dance_and_reacquires(self):
        f=Fixture()
        for i in range(220):
            d=f.step(missing=True)
            self.assertNotIn(d['spacing_phase'],('STOPPED','SEQUENCE_DONE'))
            self.assertNotEqual(d['state'],'ABORT_HOVER')
        self.assertEqual(d['state'],'HOVER_WAIT')
        self.assertEqual(dance_command(d,f.now+.015,9)[0],ZERO)
        for i in range(12):d=f.step()
        self.assertEqual(d['state'],'TRACK');self.assertGreater(d['intent']['forward'],0)

    def test_close_loss_and_clipping_keep_dance_selected(self):
        f=Fixture(width=.18)
        self.assertTrue(f.d['spacing_close_guard_armed'])
        for i in range(180):d=f.step(missing=True)
        self.assertEqual(d['spacing_phase'],'HOLD_REACQUIRE')
        window=Mock();window.dance_selected=True
        self.assertFalse(FollowLabWindow.finish_terminal_dance(window,d))
        self.assertTrue(window.dance_selected)
        d=f.step(width=.30,cx=.90)
        self.assertNotEqual(d['spacing_phase'],'STOPPED')

    def test_distance_hold_preserves_yaw_and_can_clear_close_flag(self):
        f=Fixture(width=.26)
        for i in range(10):d=f.step(width=.26,cx=.65)
        self.assertIn(d['spacing_phase'],('VISUAL_HOLD','CONFIRM_STOP'))
        self.assertGreater(d['intent']['yaw'],0);self.assertEqual(d['intent']['forward'],0)
        for i in range(70):d=f.step(width=.07,cx=.55)
        self.assertFalse(d['spacing_close_guard_armed'])
        self.assertEqual(d['spacing_phase'],'APPROACH');self.assertGreater(d['intent']['forward'],0)

    def test_stale_video_is_wait_not_search_or_terminal(self):
        f=Fixture(move=.016)
        d=f.step(missing=True,stale=True,yaw=0)
        self.assertEqual(d['state'],'WAIT_VIDEO');self.assertEqual(dance_command(d,f.now+.015,9)[0],ZERO)
        for i in range(10):d=f.step(cx=.75)
        self.assertEqual(d['state'],'TRACK')

    def test_duplicate_measurements_cannot_complete_reacquire(self):
        f=Fixture()
        for i in range(60):f.step(missing=True)
        d=f.step()
        t=f.now
        for i in range(5):
            d=f.c.tick(t+.02+i*.015,1280,720,t+.02+i*.015,f.s)
        self.assertNotEqual(d['state'],'TRACK')

    def test_near_loss_blocks_translation_but_allows_directional_search(self):
        f=Fixture(width=.17,move=.016)
        self.assertTrue(f.d['spacing_close_guard_armed'])
        for i in range(5):d=f.step(missing=True,yaw=0)
        self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
        self.assertEqual(dance_command(d,f.now+.015,9)[0],(1.,0.,0.,0.))

    def test_rejected_distractor_does_not_continue_translation(self):
        f=Fixture();t=f.now+.033
        f.c.observe([Detection(Box(0,0,12,12),.99)],t,t+.01,500,(1280,720),f.s)
        d=f.c.tick(t+.01,1280,720,t,f.s)
        self.assertFalse(d['prediction_forward_allowed'])
        self.assertEqual(dance_command(d,t+.01,9)[0],ZERO)


class SearchTests(unittest.TestCase):
    def start(self,side=1,angle=90,duration=4):
        f=Fixture(move=.016,side=side,settings=Settings(stop_width=.24,search_yaw_degrees=angle,edge_search_seconds=duration))
        for i in range(5):d=f.step(missing=True,yaw=170 if side>0 else -170)
        return f,d

    def test_motion_before_edge_is_enough_both_directions(self):
        for side in (-1,1):
            f,d=self.start(side)
            self.assertTrue(d['edge_search']['active']);self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
            self.assertEqual(dance_command(d,f.now+.015,9)[0],(float(side),0.,0.,0.))
            self.assertLess(d['edge_search']['evidence']['estimated_exit_seconds'],1.5)

    def test_measured_angle_wraparound_and_no_restart(self):
        for side in (-1,1):
            f,d=self.start(side,angle=30)
            d=f.step(missing=True,yaw=-175 if side>0 else 175)
            self.assertTrue(d['edge_search']['active'])
            self.assertAlmostEqual(d['edge_search']['angle_progress_degrees'],15)
            d=f.step(missing=True,yaw=-155 if side>0 else 155)
            self.assertFalse(d['edge_search']['active']);self.assertEqual(d['reason'],'search_angle_reached')
            for i in range(150):d=f.step(missing=True,yaw=0)
            self.assertFalse(d['edge_search']['active']);self.assertEqual(d['state'],'HOVER_WAIT')

    def test_time_limit_2_and_10_seconds(self):
        for duration in (2.,10.):
            f,d=self.start(duration=duration)
            deadline=d['edge_search']['until'];start=d['edge_search']['started']
            while f.now+.015<deadline-.05:
                d=f.step(missing=True,yaw=170)
                self.assertEqual(d['edge_search']['until'],deadline)
            while f.now+.015<deadline+.05:d=f.step(missing=True,yaw=170)
            self.assertEqual(d['state'],'HOVER_WAIT');self.assertFalse(d['edge_search']['active'])
            self.assertEqual(d['edge_search']['reason'],'search_timeout')
            self.assertAlmostEqual(deadline-start,duration)

    def test_zero_degrees_disables_yaw_search(self):
        f,d=self.start(angle=0)
        self.assertFalse(d['edge_search']['active']);self.assertEqual(yaw_override(d,f.now+.015),0)

    def test_heading_absence_never_fakes_degrees(self):
        f=Fixture(move=.016)
        for i in range(25):d=f.step(missing=True)
        self.assertEqual(d['state'],'HOVER_WAIT');self.assertEqual(d['edge_search']['reason'],'waiting_heading')
        self.assertEqual(dance_command(d,f.now+.015,9)[0],ZERO)

    def test_heading_dropout_or_source_change_ends_arc(self):
        f,d=self.start();d=f.step(missing=True)
        self.assertEqual(d['reason'],'heading_unavailable');self.assertFalse(d['edge_search']['active'])
        f,d=self.start();h=heading(f.now+.05,175);h['source']='CompassHeading'
        d=f.c.tick(f.now+.05,1280,720,f.now+.05,f.s,h)
        self.assertEqual(d['reason'],'heading_source_changed')

    def test_first_real_detection_ends_full_yaw_before_confirmation(self):
        f,d=self.start();d=f.step(cx=.75)
        self.assertFalse(d['edge_search']['active']);self.assertEqual(d['state'],'REACQUIRE')
        self.assertEqual(dance_command(d,f.now+.015,9)[0][2:],(0.,0.))

    def test_send_time_checks_deadline_heading_and_new_activation(self):
        f,d=self.start();now=f.now+.015
        self.assertNotEqual(yaw_override(d,now,9),0)
        self.assertEqual(yaw_override(d,now,now-.01),0)
        for changes in ({'state':'WAIT_VIDEO'},{'accepted':True},{'stale':True}):
            self.assertEqual(yaw_override({**d,**changes},now,9),0)
        for changes in ({'heading_time':now-1},{'angle_progress_degrees':90},
                        {'until':now-.001},{'duration_seconds':11},{'source_time':float('nan')}):
            self.assertEqual(yaw_override({**d,'edge_search':{**d['edge_search'],**changes}},now,9),0)


class KalmanTests(unittest.TestCase):
    def test_tiny_boxes_and_predictions_serialize_without_numpy_booleans(self):
        c=FollowController();s=Settings()
        for i in range(30):
            stamp=10+i/30
            ds=[Detection(Box(900+i,400,913+i,409),.9)] if i<20 else []
            c.observe(ds,stamp,stamp+.01,i+1,(1920,1080),s)
            json.dumps(c.tick(stamp+.01,1920,1080,stamp,s),allow_nan=False)

    def moving(self):
        t=BBoxTracker((1280,720))
        for i in range(20):
            stamp=10+i/30
            t.update([Detection(Box(300+i*4,300,400+i*4,350),.9)],stamp,stamp+.01,i+1)
        return t

    def test_prediction_and_covariance_do_not_create_measurements(self):
        t=self.moving();post=t.kf.statePost.copy();cov=t.kf.errorCovPost.copy();stamp=t.measurement_time
        first=t.uncertainty(stamp+.05)['position_std_px']
        for i in range(1,31):t.snapshot(stamp+i/60);t.uncertainty(stamp+i/60)
        np.testing.assert_array_equal(post,t.kf.statePost);np.testing.assert_array_equal(cov,t.kf.errorCovPost)
        self.assertEqual(t.measurement_time,stamp)
        self.assertTrue(all(a<b for a,b in zip(first,t.uncertainty(stamp+.5)['position_std_px'])))

    def test_stable_prediction_does_not_decay_after_only_100ms(self):
        t=self.moving();a=t.anchor();k=t.snapshot(a['time']+.2)
        self.assertTrue(t.quality()['stable'])
        self.assertAlmostEqual(k.vx,a['vx'],places=6)
        self.assertAlmostEqual(k.cx,a['cx']+.2*a['vx'],places=6)

    def test_forecast_is_bounded_and_filter_covariance_stays_positive(self):
        t=self.moving();a=t.anchor();k=t.snapshot(a['time']+10)
        self.assertLessEqual(abs(k.cx-a['cx']),a['limit_x'])
        self.assertTrue(np.all(np.linalg.eigvalsh(t.kf.errorCovPost)>=-1e-8))
        self.assertTrue(1<=t.process_scale<=9)

    def test_far_outlier_and_duplicate_result_are_not_evidence(self):
        t=self.moving();stamp=t.measurement_time
        self.assertFalse(t.update([Detection(Box(1100,10,1190,70),.99)],stamp+.033,stamp+.05,40))
        self.assertFalse(t.update([Detection(Box(360,300,460,350),.9)],stamp,stamp+.05,20))
        self.assertEqual(t.measurement_time,stamp)


class SettingsAndTelemetryTests(unittest.TestCase):
    def test_ranges_migration_and_saved_values(self):
        for degrees in (0.,180.):
            for seconds in (2.,10.):replace(Settings(),search_yaw_degrees=degrees,edge_search_seconds=seconds).validate()
        for change in ({'search_yaw_degrees':181},{'search_yaw_degrees':-1},{'edge_search_seconds':1.9},
                       {'edge_search_seconds':10.1},{'search_yaw_degrees':float('nan')}):
            with self.assertRaises(ValueError):replace(Settings(),**change).validate()
        old={'schema_version':4,'settings':{'stop_width':.24,'confidence':.75,'edge_search_seconds':1.,'follow_and_hold':False}}
        s=settings_from_config(old)
        self.assertEqual((s.search_yaw_degrees,s.edge_search_seconds),(90,4))
        self.assertTrue(s.follow_and_hold);self.assertEqual((s.stop_width,s.confidence),(.24,.75))
        self.assertEqual(old['settings']['edge_search_seconds'],1.)
        new=settings_from_config({'schema_version':5,'settings':asdict(replace(s,search_yaw_degrees=45,edge_search_seconds=7))})
        self.assertEqual((new.search_yaw_degrees,new.edge_search_seconds),(45,7))

    def test_heading_parser_units_and_invalid_values(self):
        for value in ('Attitude{pitch=0.0, roll=1, yaw=-179.5}','{"pitch":0,"yaw":-179.5,"roll":0}'):
            self.assertEqual(parse_heading('FlightController AircraftAttitude '+value),(-179.5,'AircraftAttitude'))
        self.assertEqual(parse_heading('FlightController CompassHeading 90.0'),(90.,'CompassHeading'))
        for value in ('FlightController AircraftAttitude null','FlightController CompassHeading NaN',
                      'FlightController CompassHeading 181','Gimbal AircraftAttitude yaw=20'):
            self.assertIsNone(parse_heading(value))
        self.assertFalse(fresh_heading(heading(11),10));self.assertFalse(fresh_heading(heading(9),10))

    def test_telemetry_socket_uses_read_only_protocol_and_handles_split_reply(self):
        client,server=socket.socketpair();requests=[]
        def serve():
            try:
                with server.makefile('rb') as stream:
                    for line in stream:
                        requests.append(line.decode().strip())
                        server.sendall(b'FlightController AircraftAttitude {yaw=')
                        server.sendall(b'12.5, pitch=0, roll=0}\r\n')
            except OSError:pass
            finally:server.close()
        worker=threading.Thread(target=serve,daemon=True);worker.start()
        def connect(address,timeout):
            self.assertEqual(address,('127.0.0.1',9997));return client
        r=HeadingReceiver('127.0.0.1',connector=connect);r.start()
        try:
            wait_for(lambda:r.get() is not None)
            self.assertEqual(r.get()['yaw_deg'],12.5)
            self.assertTrue(all(x=='get FlightController AircraftAttitude' for x in requests))
        finally:r.stop();worker.join(1)
        self.assertFalse(r.thread.is_alive())

    def test_keyboard_mapping_and_defaults_preserved(self):
        self.assertEqual(movement({'w','d','up','right'}),(1.,1.,1.,1.))
        self.assertEqual(DEFAULT_AXIS_LIMITS,dict(yaw=.5,vertical=.5,roll=.015,forward=.015))
        for code,key in ((87,'w'),(83,'s'),(65,'a'),(68,'d')):
            self.assertEqual(normalized_event_key(Mock(keysym='hebrew',keycode=code),'win32'),key)


class WireTests(unittest.TestCase):
    def test_actual_wire_search_caps_reacquire_manual_override_and_release(self):
        client,server=socket.socketpair();commands=[];events=[]
        def serve():
            try:
                with server.makefile('rb') as stream:
                    for line in stream:
                        commands.append(line.decode().strip());server.sendall(b'success\r\n')
            except OSError:pass
            finally:server.close()
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        mode={'value':'search','start':None}
        def provider():
            now=time.monotonic()
            if mode['start'] is None:mode['start']=now
            d=dict(prediction_time=now,stale=False,confirmed=True,accepted=True,
                   measurement_age_ms=20.,state='TRACK',spacing_phase='APPROACH',
                   intent=dict(yaw=.7,vertical=.55,roll=.45,forward=.3),
                   intent_basis=dict(yaw=.7,vertical=.55,roll=.45,forward=.3))
            if mode['value']=='search':
                d.update(state='DIRECTIONAL_SEARCH',accepted=False,spacing_phase='HOLD_REACQUIRE',
                  edge_search=dict(active=True,direction=1,started=mode['start'],until=mode['start']+4,
                    source_time=mode['start']-.15,heading_time=now,angle_degrees=90,
                    angle_progress_degrees=10,duration_seconds=4))
            elif mode['value']=='hover':d.update(state='HOVER_WAIT',reason='search_timeout')
            return d
        c=ManualControl('127.0.0.1',connector=lambda *a,**kw:client,decision_provider=provider,
                        on_event=lambda e,d:events.append((e,d)))
        heartbeat_stop=threading.Event();keys=set()
        def beat():
            while not heartbeat_stop.wait(.03):c.update(keys)
        heartbeat=threading.Thread(target=beat,daemon=True)
        try:
            c.set_axis_limits(dict(yaw=.05,vertical=.50,roll=.015,forward=.015))
            c.start();c.update(set());c.enable();wait_for(lambda:c.enabled)
            c.start_dance();mode['start']=time.monotonic()+.001;heartbeat.start()
            wait_for(lambda:'rc 1.0000 0.0000 0.0000 0.0000' in commands)
            keys.add('a');wait_for(lambda:'rc -0.0500 0.0000 0.0000 0.0000' in commands)
            keys.clear();mode['value']='hover'
            wait_for(lambda:c.dance_status.startswith('Hover /'))
            self.assertEqual(c.mode,'DANCE');self.assertTrue(c.wanted)
            mode['value']='track'
            wait_for(lambda:any(e=='flight_command' and d.get('mode')=='DANCE' and d.get('values',[0]*4)[3]>0 for e,d in events))
            for e,d in events:
                if e!='flight_command' or d.get('command')!='rc':continue
                limits=(1.,0.,0.,0.) if d['edge_yaw_override'] else (.05,.5,.015,.015)
                self.assertTrue(all(abs(v)<=limit+1e-8 for v,limit in zip(d['values'],limits)))
            c.release();wait_for(lambda:'disable' in commands)
            self.assertEqual(commands[commands.index('disable')-1],'rc 0 0 0 0')
        finally:
            heartbeat_stop.set();c.stop();c.thread.join(2);thread.join(1)


if __name__=='__main__':unittest.main(verbosity=2)
