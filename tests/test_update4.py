"""Regression cases derived from the 20260922_180615_a9785b session."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import unittest
from dataclasses import replace
from unittest.mock import Mock
import tkinter as tk

from follow_neo.dance import DanceSmoother, dance_command, ZERO
from follow_neo.gui import FollowLabWindow, safe_focus_widget
from follow_neo.controller import FollowController
from follow_neo.spacing import VisualSpacingController
from follow_neo.types import Box, Detection, Intent, Settings
from follow_neo.manual import ManualControl, shape_dance_axis
from test_dance import decision
from test_manual import Server, wait_for


class SmoothTests(unittest.TestCase):
    def test_start_and_reversal_have_bounded_rate(self):
        s=DanceSmoother(); previous=ZERO
        for i in range(30):
            target=(1.,1.,1.,1.) if i<12 else (-1.,-1.,-1.,-1.)
            out=s.update(target,10+i*.05)
            for j in range(3): self.assertLessEqual(abs(out[j]-previous[j]),s.RATES[j]*.05+1e-9)
            self.assertTrue(all(abs(v)<=1 for v in out))
            previous=out

    def test_stop_and_bbox_deceleration_never_wait_for_ramp(self):
        s=DanceSmoother()
        for i in range(20): s.update((1.,1.,1.,1.),10+i*.05)
        self.assertEqual(s.update((1.,1.,1.,.03),11)[3],.03)
        self.assertEqual(s.update((1.,1.,1.,0.),11.05)[3],0.)
        self.assertEqual(s.update((1.,1.,1.,1.),11.1,active=False),ZERO)

    def test_low_caps_are_continuous_at_three_percent_boundary(self):
        for cap in (.015,.02999,.03,.03001,.5):
            self.assertEqual(shape_dance_axis(.2,cap),.2)
            self.assertEqual(shape_dance_axis(0,cap),0.)

    def test_latest_decision_provider_works_without_gui_decision_copy(self):
        server=Server(); provider=Mock(side_effect=lambda:decision())
        c=ManualControl('127.0.0.1',server.connect,decision_provider=provider)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.set_axis_limits(dict(yaw=.5,vertical=.5,roll=.015,forward=.015))
        c.start(); c.update(set()); c.enable()
        wait_for(lambda:c.enabled); c.start_dance()
        wait_for(lambda:any(cmd.startswith('rc ') and float(cmd.split()[-1])>0 for cmd in server.commands))
        self.assertIsNone(c.decision)


class ContinuityTests(unittest.TestCase):
    def setup_track(self,width=100):
        c=FollowController(); s=Settings(stop_width=.38)
        detections=[Detection(Box(600,310,600+width,390),.9)]
        for i in range(6):
            t=10+i*.05
            c.observe(detections,t,t+.02,i+1,(1280,720),s)
            d=c.tick(t+.02,1280,720,t+.02,s)
        self.assertEqual(d['state'],'TRACK')
        return c,s,d

    def test_single_empty_detection_bridges_without_refreshing_measurement_time(self):
        c,s,d=self.setup_track()
        c.observe([],10.30,10.32,7,(1280,720),s)
        gap=c.tick(10.32,1280,720,10.32,s)
        self.assertFalse(gap['accepted']); self.assertTrue(gap['brief_detection_gap'])
        self.assertEqual(gap['measurement_time'],10.25)
        self.assertGreater(gap['intent']['forward'],0)
        self.assertGreater(dance_command(gap,10.33,9)[0][3],0)
        expired=c.tick(10.51,1280,720,10.51,s)
        self.assertFalse(expired['brief_detection_gap'])
        # Update 6 adds a separate stable-forward contract. The original brief
        # bridge must still expire without that authorization.
        expired['prediction_forward_allowed']=False
        self.assertEqual(dance_command(expired,10.51,9)[0],ZERO)

    def test_bridge_decays_and_cannot_be_kept_alive_by_repeated_predictions(self):
        c,s,_=self.setup_track()
        c.observe([],10.30,10.32,7,(1280,720),s)
        gap=c.tick(10.40,1280,720,10.40,s)
        gap['prediction_forward_allowed']=False
        a=dance_command(gap,10.40,9)[0][3]
        b=dance_command(gap,10.48,9)[0][3]
        self.assertGreater(a,b); self.assertGreater(b,0)
        self.assertEqual(dance_command(gap,10.51,9)[0],ZERO)

    def test_rejected_competing_detection_is_not_bridged(self):
        c,s,_=self.setup_track()
        c.observe([Detection(Box(0,0,20,20),.99)],10.30,10.32,7,(1280,720),s)
        d=c.tick(10.32,1280,720,10.32,s)
        self.assertFalse(d['brief_detection_gap'])
        self.assertEqual(dance_command(d,10.33,9)[0],ZERO)

    def test_near_target_miss_holds_zero_not_blind_forward(self):
        c,s,_=self.setup_track(width=320)
        c.observe([],10.30,10.32,7,(1280,720),s)
        d=c.tick(10.32,1280,720,10.32,s)
        self.assertFalse(d['brief_detection_gap'])
        self.assertEqual(d['spacing_phase'],'TRACK_PAUSE')
        self.assertEqual(d['intent'],vars(Intent()))

    def test_stale_video_still_stops(self):
        c,s,_=self.setup_track()
        c.observe([],10.30,10.32,7,(1280,720),s)
        d=c.tick(10.32,1280,720,9,s)
        self.assertFalse(d['brief_detection_gap']); self.assertTrue(d['stale'])
        self.assertEqual(dance_command(d,10.33,9)[0],ZERO)

    def test_pause_phases_cannot_send_intent_or_claim_active_tracking(self):
        for phase in ('TRACK_PAUSE','CONFIRM_STOP'):
            d=decision(); d['spacing_phase']=phase
            values,status=dance_command(d,time.monotonic(),0)
            self.assertEqual(values,ZERO)
            self.assertFalse(status.startswith('Tracking /'))


class SpacingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.p=VisualSpacingController(); self.s=Settings(stop_width=.38)

    def sample(self,t,mid,width,**kwargs):
        args=dict(now=t,track_active=True,valid=True,rejected=False,clipped=False,
                  timed_out=False,mid=mid,mtime=t-.047,age=.047,raw_width=width,
                  filtered_width=width,ex=0,ey=0,command=Intent(forward=.7),settings=self.s)
        args.update(kwargs)
        return self.p.update(**args)

    def test_logged_width_spike_recovers_after_distinct_clear_frames(self):
        self.sample(10.,1,.11965)
        self.sample(10.031,2,.23236)
        self.assertEqual(self.p.phase,'CONFIRM_STOP')
        cmd,_=self.sample(10.063,3,.12268)
        self.assertEqual(cmd,Intent()); self.assertEqual(self.p.phase,'CONFIRM_STOP')
        self.sample(10.11,4,.120)
        self.sample(10.18,5,.121)
        self.assertEqual(self.p.phase,'APPROACH'); self.assertFalse(self.p.stopped_now)

    def test_duplicate_measurement_never_counts_for_resume(self):
        self.sample(10.,1,.11965); self.sample(10.031,2,.23236)
        self.sample(10.063,3,.12268)
        for t in (10.10,10.13,10.18):
            self.sample(t,3,.12268,mtime=10.016,age=t-10.016)
        self.assertEqual(self.p.phase,'CONFIRM_STOP'); self.assertEqual(self.p.resume_hits,1)

    def test_close_miss_recovers_but_long_loss_is_terminal(self):
        self.sample(10,1,.25); self.sample(10.05,2,.25,rejected=True)
        self.assertEqual(self.p.phase,'TRACK_PAUSE')
        self.sample(10.10,3,.25); self.sample(10.16,4,.25); self.sample(10.22,5,.25)
        self.assertEqual(self.p.phase,'APPROACH')
        self.sample(10.25,6,.25,rejected=True)
        cmd,_=self.sample(11.,7,.25,rejected=True)
        self.assertEqual(self.p.phase,'STOPPED'); self.assertEqual(cmd,Intent())
        self.sample(11.1,8,.20)
        self.assertEqual(self.p.phase,'STOPPED')

    def test_real_bbox_threshold_still_stops_forward(self):
        self.sample(10,1,.39)
        cmd,_=self.sample(10.1,2,.39)
        self.assertLessEqual(cmd.forward,0)
        self.assertIn(self.p.phase,('VISUAL_HOLD','BRAKE'))

    def test_one_shot_retains_terminal_false_threshold_contract(self):
        self.s=replace(self.s,follow_and_hold=False)
        self.sample(10.,1,.11965); self.sample(10.031,2,.23236)
        self.sample(10.063,3,.12268)
        self.assertEqual(self.p.phase,'STOPPED')


class GuiRegressionTests(unittest.TestCase):
    def test_native_popdown_resolves_to_combobox(self):
        root=Mock(); combo=object()
        root.focus_get.side_effect=KeyError('popdown')
        root.tk.call.return_value='.frame.combo.popdown.f.l'
        root.nametowidget.side_effect=lambda name: combo if name=='.frame.combo' else (_ for _ in ()).throw(KeyError(name))
        self.assertIs(safe_focus_widget(root),combo)

    def test_destroyed_root_does_not_raise(self):
        root=Mock(); root.focus_get.side_effect=tk.TclError('destroyed')
        self.assertIsNone(safe_focus_widget(root))

    def test_recording_toolbar_does_not_release_dance(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.root=Mock(); w.canvas=object(); w.view_bar=object(); w.pressed={'w'}
        w.manual=ManualControl('127.0.0.1'); w.manual.wanted=True; w.manual.start_dance()
        w.root.focus_get.return_value=Mock(master=w.view_bar)
        w.focus_out(Mock()); w.root.after_idle.call_args.args[0]()
        self.assertTrue(w.manual.wanted); self.assertEqual(w.pressed,set())

    def test_incomplete_numeric_entry_preserves_previous_limit(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.manual=ManualControl('127.0.0.1'); w.notice=Mock()
        w.axis_limit_vars={'forward':Mock()}; w.axis_entry_vars={'forward':Mock()}
        w.axis_limit_labels={'forward':Mock()}
        w.change_axis_speed('forward','')
        self.assertEqual(w.manual.axis_limits['forward'],.015)
        w.axis_entry_vars['forward'].set.assert_called_with('1.5')

    def test_focus_callback_is_not_scheduled_after_close(self):
        w=FollowLabWindow.__new__(FollowLabWindow); w.root=Mock(); w.closing=True
        w.focus_out(Mock()); w.root.after_idle.assert_not_called()


if __name__=='__main__': unittest.main()
