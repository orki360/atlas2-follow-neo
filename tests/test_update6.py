"""Timing, noisy-box, bounded continuation and actual transport regressions."""
import sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import numpy as np
from follow_neo.tracker import BBoxTracker
from follow_neo.types import Box,Detection,Settings
from follow_neo.controller import FollowController
from follow_neo.dance import dance_command,ZERO
from follow_neo.overlay import box_at_frame
from follow_neo.video import Frame
from follow_neo.manual import ManualControl
from test_manual import Server,wait_for


def measurement(cx,cy=220,w=80,h=50):
    return [Detection(Box(cx-w/2,cy-h/2,cx+w/2,cy+h/2),.8)]


class TimingAndNoiseTests(unittest.TestCase):
    def test_filter_does_not_depend_on_latency_or_display_polling(self):
        a=BBoxTracker();b=BBoxTracker()
        for i in range(30):
            stamp=10+i/30;boxes=measurement(200+i*3)
            a.update(boxes,stamp,stamp+.025,i+1)
            b.update(boxes,stamp,stamp+(.14 if i%2 else .06),i+1)
            for future in (stamp+.016,stamp+.2,stamp+.4):b.snapshot(future)
            np.testing.assert_allclose(a.kf.statePost,b.kf.statePost,atol=1e-10)
            np.testing.assert_allclose(a.kf.errorCovPost,b.kf.errorCovPost,atol=1e-10)
            self.assertEqual(b.last_time,stamp)

    def test_prediction_reads_never_change_measurement_evidence(self):
        t=BBoxTracker()
        for i in range(8):t.update(measurement(220+i*3),10+i*.04,10.02+i*.04,i+1)
        before=t.kf.statePost.copy();quality=t.quality();mtime=t.measurement_time
        for when in np.linspace(10.3,11.2,80):t.snapshot(when)
        np.testing.assert_array_equal(t.kf.statePost,before)
        self.assertEqual(t.quality(),quality);self.assertEqual(t.measurement_time,mtime)

    def test_stationary_jitter_does_not_create_large_velocity(self):
        t=BBoxTracker();speeds=[];errors=[]
        for i in range(45):
            stamp=10+i/30
            t.update(measurement(300+8*(-1)**i),stamp,stamp+.07,i+1)
            k=t.snapshot(stamp+.07)
            if i>15:speeds.append(abs(k.vx));errors.append(abs(k.cx-300))
        self.assertLess(max(speeds),50)
        self.assertLess(max(errors),7)
        self.assertLess(abs(t.snapshot(stamp+.4).cx-300),16)

    def test_constant_motion_stays_responsive_despite_box_noise(self):
        t=BBoxTracker()
        for i in range(45):
            stamp=10+i/30
            t.update(measurement(180+100*i/30+8*(-1)**i),stamp,stamp+.06,i+1)
        k=t.snapshot(stamp+.06)
        self.assertLess(abs(k.cx-(180+100*(stamp+.06-10))),7)
        self.assertGreater(k.vx,50);self.assertLess(k.vx,150)

    def test_loss_displacement_is_bounded_and_speed_decays(self):
        t=BBoxTracker()
        for i in range(20):t.update(measurement(100+i*8),10+i*.03,10.02+i*.03,i+1)
        a=t.snapshot(t.measurement_time);speeds=[]
        for age in (.12,.2,.35,.5,.65,2):
            k=t.snapshot(t.measurement_time+age)
            self.assertLessEqual(abs(k.cx-a.cx),a.width*.8+1e-9)
            self.assertAlmostEqual(k.width,a.width)
            self.assertGreaterEqual(k.vx,0);speeds.append(k.vx)
        self.assertEqual(speeds,sorted(speeds,reverse=True))

    def test_size_change_does_not_invent_center_motion(self):
        t=BBoxTracker()
        for i in range(20):t.update(measurement(300,w=80+i*2,h=50+i),10+i*.03,10.06+i*.03,i+1)
        k=t.snapshot(10.9)
        self.assertAlmostEqual(k.cx,300);self.assertAlmostEqual(k.vx,0)

    def test_inconsistent_boxes_disallow_extended_prediction(self):
        t=BBoxTracker()
        for i in range(9):t.update(measurement(300+25*(-1)**i),10+i*.03,10.05+i*.03,i+1)
        self.assertFalse(t.quality()['stable'])
        self.assertEqual(t.quality()['reason'],'inconsistent_boxes')

    def test_weak_boxes_do_not_grant_extended_forward(self):
        t=BBoxTracker()
        for i in range(9):
            boxes=measurement(300)
            if i>1: boxes=[Detection(boxes[0].box,.26)]
            t.update(boxes,10+i*.03,10.05+i*.03,i+1,.3,.3,.25)
        self.assertFalse(t.quality()['stable'])


class ForwardContinuationTests(unittest.TestCase):
    def ready(self,jitter=0):
        c=FollowController();s=Settings(confidence=.3,stop_width=.38)
        for i in range(8):
            stamp=10+i*.04
            c.observe(measurement(680+jitter*(-1)**i,330,100,60),stamp,stamp+.02,i+1,(1280,720),s)
            d=c.tick(stamp+.02,1280,720,stamp+.02,s)
        self.assertEqual(d['state'],'TRACK')
        return c,s

    def gap(self,c,s,age):
        now=10.28+age
        c.observe([],now-.02,now,100+round(age*1000),(1280,720),s)
        return c.tick(now,1280,720,now,s)

    def test_stable_forward_decays_smoothly_across_old_250ms_cutoff(self):
        c,s=self.ready();previous=1.
        for age in (.10,.15,.23,.249,.251,.3,.4,.451):
            d=self.gap(c,s,age);values,_=dance_command(d,d['prediction_time'],9)
            self.assertLessEqual(values[3],previous+1e-9);previous=values[3]
            self.assertEqual(d['measurement_time'],10.28);self.assertFalse(d['accepted'])
            if .25<age<.45:
                self.assertGreater(values[3],0);self.assertEqual(values[2],0)
            if age>=.45:self.assertEqual(values[3],0)

    def test_no_extension_for_unstable_history(self):
        c,s=self.ready(jitter=40);d=self.gap(c,s,.35)
        self.assertFalse(d['prediction_quality']['stable'])
        self.assertFalse(d['prediction_forward_allowed'])
        self.assertEqual(dance_command(d,d['prediction_time'],9)[0][3],0)

    def test_send_time_expiry_even_for_frozen_recent_decision(self):
        c,s=self.ready();d=self.gap(c,s,.38);stamp=d['prediction_time']
        self.assertGreater(dance_command(d,stamp,9)[0][3],0)
        self.assertEqual(dance_command(d,stamp+.08,9)[0][3],0)
        self.assertEqual(dance_command(d,stamp+.201,9)[0],ZERO)

    def test_empty_frames_cannot_refresh_prediction_deadline(self):
        c,s=self.ready()
        for age in (.3,.4,.5,.6,.7):
            d=self.gap(c,s,age)
            if age>=.45:self.assertEqual(dance_command(d,d['prediction_time'],9)[0][3],0)
        self.assertIsNone(d['track_box'])

    def test_real_reacquisition_replaces_prediction_support(self):
        c,s=self.ready();self.gap(c,s,.32)
        for i in range(3):
            now=10.65+i*.04
            c.observe(measurement(683,330,100,60),now,now+.02,500+i,(1280,720),s)
            d=c.tick(now+.02,1280,720,now+.02,s)
        self.assertTrue(d['accepted']);self.assertEqual(d['track_support'],'yolo')
        self.assertFalse(d['prediction_forward_allowed']);self.assertEqual(d['state'],'TRACK')

    def test_stop_threshold_and_stale_override_extended_forward(self):
        c,s=self.ready();d=self.gap(c,s,.35)
        for change in ({'spacing_phase':'STOPPED'},{'spacing_phase':'TRACK_PAUSE'},
                       {'spacing_phase':'CONFIRM_STOP'},{'bbox_clipped':True},
                       {'spacing_close_guard_armed':True},{'stale':True}):
            self.assertEqual(dance_command({**d,**change},d['prediction_time'],9)[0],ZERO)

    def test_predicted_overlay_uses_same_anchor_not_backwards_linear_velocity(self):
        c,s=self.ready()
        for i in range(8):
            stamp=10.32+i*.03
            c.observe(measurement(680+i*5,330,100,60),stamp,stamp+.06,100+i,(1280,720),s)
        d=c.tick(stamp+.20,1280,720,stamp+.20,s)
        frame=Frame(999,stamp+.16,np.zeros((720,1280,3),np.uint8))
        box=box_at_frame(frame,d,stamp+.20);k=c.tracker.snapshot(frame.decoded_at)
        self.assertIsNotNone(box)
        self.assertAlmostEqual((box['x1']+box['x2'])/2,k.cx)
        self.assertAlmostEqual((box['y1']+box['y2'])/2,k.cy)

    def test_transmitted_forward_stays_below_gui_cap_during_extension(self):
        c,s=self.ready();template=self.gap(c,s,.30)
        server=Server();events=[];state={'age':300}
        def provider():
            now=time.monotonic()
            return {**template,'prediction_time':now,'measurement_age_ms':state['age']}
        control=ManualControl('127.0.0.1',server.connect,decision_provider=provider,
                              on_event=lambda event,data:events.append((event,data)))
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.5,vertical=.5,roll=.015,forward=.015))
        control.start();control.update(set());control.enable();wait_for(lambda:control.enabled)
        control.start_dance()
        wait_for(lambda:any(e=='flight_command' and d.get('prediction_forward_allowed') and d['values'][3]>0 for e,d in events))
        positive=[d for e,d in events if e=='flight_command' and d.get('prediction_forward_allowed') and d['values'][3]>0]
        for d in positive:
            self.assertLessEqual(d['values'][3],.015)
            self.assertEqual(d['values'][2],0.)
        state['age']=451;control.update(set());start=len(events)
        wait_for(lambda:any(e=='flight_command' and d['values'][3]==0 for e,d in events[start:]))


if __name__=='__main__':unittest.main()
