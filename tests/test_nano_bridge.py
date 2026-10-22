"""Offline invariants: visual continuity must never manufacture NEO evidence."""
import copy
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from follow_neo.control.nano_guard import apply_visual, guard_reason, strong_anchor, transport_visual_reason
from follow_neo.perception.nano_tracker import NanoWorker, AppearanceCheck, TemplateRefresh, historical_box
from follow_neo.capture.video import Frame
from follow_neo.shared.types import Box, Detection, Settings
from follow_neo.control.dance import dance_command, ZERO
from follow_neo.transport.control_transport import ControlChannel, VISUAL_CHANGE
from follow_neo.presentation.overlay import render_bundle


def visual(now=10., **changes):
    v=dict(mode='guarded',blocked=False,reason='visual_supported',epoch=2,generation=1,identity=7,
           semantic_time=now-.12,visual_time=now-.02,deadline=now+.08,
           box=[110.,40.,140.,60.],frame_id=15)
    v.update(changes);return v


def decision(now=10., **changes):
    d=dict(prediction_time=now,measurement_time=now-.12,measurement_age_ms=120.,measurement_id=12,
           last_strong_measurement_time=now-.12,confirmed=True,accepted=True,state='TRACK',stale=False,
           spacing_phase='APPROACH',frame_size=[200,100],selected_class_id=0,selected_class_name='neo',
           target_identity=dict(logical_target_id=7),class_support=None,
           intent=dict(yaw=.1,vertical=.1,roll=.1,forward=.1),
           intent_basis=dict(yaw=.7,vertical=.55,roll=.45,forward=.3),reacquire_hits=3)
    d.update(changes);return d


class NanoGuardTests(unittest.TestCase):
    def test_shadow_does_not_change_control_or_identity(self):
        d=decision();out=apply_visual(d,visual(mode='shadow',blocked=True),10.)
        for k,v in d.items():self.assertEqual(out[k],v)
        self.assertEqual(dance_command(out,10.,9.)[0],dance_command(d,10.,9.)[0])

    def test_visual_updates_cannot_renew_semantic_evidence(self):
        d=decision();before=copy.deepcopy(d)
        out=apply_visual(d,visual(),10.)
        for key in ('measurement_time','measurement_id','last_strong_measurement_time','accepted','reacquire_hits','target_identity'):
            self.assertEqual(out[key],d[key])
        self.assertEqual(d,before)

    def test_bridge_only_yaw_even_with_old_forward_coast(self):
        d=decision(accepted=False,state='COAST',prediction_recovery=True,prediction_forward_allowed=True)
        out=apply_visual(d,visual(),10.)
        command,reason=dance_command(out,10.,9.)
        self.assertGreater(command[0],0);self.assertEqual(command[1:],(0.,0.,0.))
        self.assertFalse(out['prediction_forward_allowed']);self.assertTrue(reason.startswith('Nano bridge'))

    def test_weak_measurement_does_not_authorize_translation(self):
        d=decision(measurement_time=9.95)
        out=apply_visual(d,visual(),10.)
        self.assertEqual(dance_command(out,10.,9.)[0][1:],(0.,0.,0.))

    def test_veto_wins_over_scan_and_coast(self):
        for state in ('TRACK','COAST','DIRECTIONAL_SEARCH','REACQUIRE'):
            out=apply_visual(decision(state=state,prediction_recovery=True,prediction_forward_allowed=True),
                             visual(blocked=True,reason='background'),10.)
            self.assertEqual(dance_command(out,10.,9.)[0],ZERO)
            self.assertFalse(out['prediction_recovery']);self.assertFalse(out['prediction_forward_allowed'])

    def test_bad_times_and_expiry_fail_closed(self):
        for v in (visual(semantic_time=9.599),visual(visual_time=9.899),visual(deadline=9.99),
                  visual(semantic_time=10.01),visual(visual_time=float('nan')),visual(deadline=None)):
            self.assertIsNotNone(guard_reason(v,10.))
            self.assertEqual(dance_command(apply_visual(decision(accepted=False),v,10.),10.,9.)[0],ZERO)

    def test_identity_mismatch_fails_closed(self):
        out=apply_visual(decision(),visual(identity=8),10.)
        self.assertEqual(dance_command(out,10.,9.)[0],ZERO)

    def test_malformed_visual_box_cannot_become_yaw(self):
        for box in (None,[0,0,1],[0,0,float('nan'),1],[0,0,0,10]):
            self.assertIsNotNone(guard_reason(visual(box=box),10.))
        d=decision(nano=visual(),nano_authority='bridge_yaw',target_identity=None)
        self.assertEqual(dance_command(d,10.,9.)[0],ZERO)

    def test_old_decision_expires_despite_newer_worker_result(self):
        old=decision(nano=visual());new=decision(nano=visual(10.2))
        self.assertIsNotNone(transport_visual_reason(old,new,10.2))

    def test_pending_command_cancelled_on_reset_veto_or_mode_change(self):
        old=decision(nano=visual())
        for new in (decision(nano=visual(epoch=3)),decision(nano=visual(blocked=True)),
                    decision(nano=visual(mode='shadow')),None):
            self.assertIsNotNone(transport_visual_reason(old,new,10.))
        self.assertIsNone(transport_visual_reason(old,decision(nano=visual()),10.))

    def test_strong_anchor_requires_raw_neo_and_unique_association(self):
        d=dict(box=[10,10,30,30],class_id=0,class_name='neo',confidence=.8,segments=[[[10,10],[30,10],[20,30]]])
        t=SimpleNamespace(confirmed=True,accepted=Detection(Box(10,10,30,30),.8),measurement_weak=False,logical_target_id=3)
        c=SimpleNamespace(tracker=t,rejected=False,class_support=None)
        r=dict(detections=[d],frame=object(),generation=0)
        self.assertIs(strong_anchor(c,r,Settings())['detection'],d)
        c.class_support={};self.assertIsNone(strong_anchor(c,r,Settings()));c.class_support=None
        t.measurement_weak=True;self.assertIsNone(strong_anchor(c,r,Settings()));t.measurement_weak=False
        r['detections']=[dict(d,class_id=1,class_name='MINI PRO4')];self.assertIsNone(strong_anchor(c,r,Settings()))
        r['detections']=[d,dict(d,class_id=1,confidence=.82)];self.assertIsNone(strong_anchor(c,r,Settings()))

    def test_mode_validation_and_default_guarded(self):
        self.assertEqual(Settings().nano_mode,'guarded')
        with self.assertRaises(ValueError):Settings(nano_mode='anything').validate()

    def test_registered_dual_models_reach_the_final_gate(self):
        from follow_neo.perception.models import DUAL_MODELS, DUAL_MODEL_SOURCE_NAMES, target_policy
        for name in DUAL_MODELS:
            d=decision(target_policy=target_policy(name,DUAL_MODEL_SOURCE_NAMES[name]))
            self.assertTrue(any(dance_command(d,10.,9.,require_target_policy=True)[0]),name)
        d=decision(target_policy=dict(dance_allowed=True,model_name='unregistered.onnx',target_class_id=0,target_class_name='neo'))
        self.assertEqual(dance_command(d,10.,9.,require_target_policy=True)[0],ZERO)


class NanoWorkerTests(unittest.TestCase):
    def setUp(self):
        self.now=10.;self.w=NanoWorker('.',clock=lambda:self.now)
        self.image=np.zeros((100,200,3),np.uint8)

    def anchor(self, stamp, fid=1, **changes):
        return dict(frame=Frame(fid,stamp,self.image),detection=dict(box=[10,10,30,30]),identity=7,generation=0,**changes)

    def test_delayed_pre_veto_anchor_cannot_rearm(self):
        self.now=10.2;self.w.veto('background')
        self.w.submit_anchor(self.anchor(10.1))
        self.assertIsNone(self.w.pending);self.assertTrue(self.w.snapshot()['blocked'])
        self.w.submit_anchor(self.anchor(10.21,2));self.assertFalse(self.w.pending['arm'])
        self.w.submit_anchor(self.anchor(10.31,3));self.assertTrue(self.w.pending['arm'])

    def test_repeated_same_frame_does_not_vote(self):
        a=self.anchor(10.1)
        for _ in range(8):self.w.submit_anchor(a)
        self.assertEqual(len(self.w.hits),1)

    def test_skipped_anchor_frame_uses_bounded_interpolation_not_extrapolation(self):
        history={1:(10.,[0,0,20,20]),3:(10.04,[20,0,40,20])}
        box=historical_box(history,Frame(2,10.02,self.image))
        np.testing.assert_allclose(box,[10,0,30,20],atol=1e-10)
        self.assertIsNone(historical_box(history,Frame(4,10.06,self.image)))
        self.assertIsNone(historical_box({1:history[1],9:(10.3,[20,0,40,20])},Frame(4,10.06,self.image)))

    def test_reset_discards_old_generation_and_epoch(self):
        self.w.reset('guarded',1);self.w.submit_anchor(self.anchor(10.1))
        self.assertIsNone(self.w.pending)
        epoch=self.w.epoch;self.w.veto('late_worker',expected_epoch=epoch-1)
        self.assertEqual(self.w.epoch,epoch)

    def test_frame_history_is_bounded(self):
        for i in range(100):self.w.submit_frame(Frame(i,10+i/60,self.image))
        self.assertLessEqual(len(self.w.frames),20)
        self.assertLessEqual(self.w.frames[-1].decoded_at-self.w.frames[0].decoded_at,.35)

    def test_semantic_expiry_blocks_control_but_keeps_visual_track(self):
        self.w.blocked=False;self.w.semantic_time=10.01
        self.w.observation=dict(visual_time=10.5,box=[10,10,30,30])
        self.now=10.5
        snapshot=self.w.snapshot();self.assertTrue(snapshot['blocked'])
        self.assertEqual(snapshot['reason'],'semantic_expired');self.assertIn('box',snapshot)
        self.assertTrue(snapshot['visual_active']);self.assertEqual(snapshot['visual_state'],'unverified')
        self.assertEqual(self.w.epoch,0);self.assertEqual(self.w.semantic_time,10.01)

    def test_submitted_anchor_does_not_renew_semantic_clock_before_validation(self):
        self.w.submit_anchor(self.anchor(10.01))
        self.assertIsNone(self.w.snapshot()['semantic_time'])
        self.assertIsNotNone(self.w.pending)

    def test_frame_exit_veto_precedes_background_capture(self):
        check=AppearanceCheck();frame=Frame(1,10.1,self.image)
        reason,metrics=check.update(frame,[-20,10,10,30])
        self.assertEqual(reason,'target_leaving_frame')

    def test_flat_or_blurred_patch_alone_not_a_drift_verdict(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,30,30]))
        reason,_=check.update(Frame(2,10.13,self.image),[10,10,30,30])
        self.assertIsNone(reason)

    def test_appearance_change_and_stationary_motion_not_counted_twice(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,30,30]))
        check.reference=np.arange(48*32,dtype=np.float32).reshape(32,48)
        check.patch=Mock(return_value=-check.reference)
        check.flow=Mock(return_value=dict(relative_motion_px=.2,box_flow_error_px=0.))
        for i in range(4):
            reason,_=check.update(Frame(i+2,10.12+i*.02,self.image),[10,10,30,30])
            self.assertIsNone(reason)

    def test_far_target_flow_warning_blocks_control_without_destroying_track(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,52,29]))
        check.reference=np.arange(48*32,dtype=np.float32).reshape(32,48)
        check.patch=Mock(return_value=-check.reference)
        check.flow=Mock(return_value=dict(relative_motion_px=2.6,box_flow_error_px=9.,
                                          flow_points=7,foreground_flow_spread_px=1.))
        for i in range(3):
            reason,metrics=check.update(Frame(i+2,10.12+i*.03,self.image),[10,10,52,29])
            self.assertIsNone(reason)
        self.assertTrue(metrics['control_suspect'])
        self.w.blocked=False;self.w.semantic_time=9.95
        self.w.observation=dict(visual_time=9.99,box=[10,10,52,29],checks=metrics)
        result=self.w.snapshot()
        self.assertTrue(result['blocked']);self.assertEqual(result['reason'],'visual_appearance_unsupported')
        self.assertTrue(result['visual_active'])

    def test_coherent_large_flow_disagreement_still_rejects_drift(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,52,29]))
        check.reference=np.arange(48*32,dtype=np.float32).reshape(32,48)
        check.patch=Mock(return_value=-check.reference)
        check.flow=Mock(return_value=dict(relative_motion_px=6.,box_flow_error_px=30.,
                                          flow_points=9,foreground_flow_spread_px=1.))
        for i in range(2):
            reason,metrics=check.update(Frame(i+2,10.12+i*.03,self.image),[10,10,52,29])
        self.assertEqual(reason,'appearance_or_background_drift')

    def test_textured_target_replaced_by_flat_background_is_rejected(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,52,29]))
        check.reference=(np.indices((32,48)).sum(axis=0)%2*80).astype(np.float32)
        check.patch=Mock(return_value=np.full((32,48),180.,np.float32))
        check.flow=Mock(return_value={})
        for i in range(3):
            reason,metrics=check.update(Frame(i+2,10.12+i*.03,self.image),[10,10,52,29])
            if i<2:self.assertIsNone(reason)
        self.assertEqual(reason,'target_texture_lost');self.assertTrue(metrics['control_suspect'])

    def test_textured_distractor_cannot_persist_with_unmatched_appearance(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,52,29]))
        check.reference=np.arange(48*32,dtype=np.float32).reshape(32,48)
        check.patch=Mock(return_value=-check.reference);check.flow=Mock(return_value={})
        for i in range(24):
            reason,metrics=check.update(Frame(i+2,10.12+i*.03,self.image),[10,10,52,29])
        self.assertEqual(reason,'appearance_unverified');self.assertTrue(metrics['control_suspect'])

    def test_visual_quality_can_stay_valid_without_a_new_yolo_anchor(self):
        check=AppearanceCheck();f=Frame(1,10.1,self.image)
        check.anchor(f,dict(box=[10,10,52,29]))
        check.reference=np.arange(48*32,dtype=np.float32).reshape(32,48)
        check.patch=Mock(return_value=check.reference.copy());check.flow=Mock(return_value={})
        for i in range(60):
            reason,metrics=check.update(Frame(i+2,10.12+i*.03,self.image),[10,10,52,29])
            self.assertIsNone(reason)
        self.assertEqual(metrics['appearance_unmatched_ms'],0.)

    def test_worker_runs_without_reinitializing_each_intermediate_frame(self):
        tracker=Mock();tracker.update.return_value=(True,(10,10,20,20));tracker.getTrackingScore.return_value=.8
        clock=[10.];w=NanoWorker('.',factory=lambda:tracker,clock=lambda:clock[0])
        w.start();self.addCleanup(w.stop)
        def wait(predicate):
            until=time.monotonic()+1.
            while not predicate() and time.monotonic()<until:time.sleep(.001)
            self.assertTrue(predicate())
        for i in range(50):
            clock[0]+=.02;f=Frame(i,clock[0],self.image)
            w.submit_frame(f)
            if i==0:
                w.submit_anchor(dict(frame=f,detection=dict(box=[10,10,30,30]),identity=7,generation=0))
                wait(lambda:tracker.init.called)
            else:
                wait(lambda:(w.observation or {}).get('frame_id')==i)
                if i==3:
                    w.submit_anchor(dict(frame=f,detection=dict(box=[10,10,30,30]),identity=7,generation=0))
        self.assertGreaterEqual(w.updates,45);self.assertTrue(w.snapshot()['blocked'])
        self.assertEqual(w.snapshot()['reason'],'semantic_expired')
        self.assertTrue(w.snapshot()['visual_active'])
        self.assertLessEqual(tracker.init.call_count,2)

    def test_historical_box_never_uses_future_frame_or_grants_control(self):
        self.w.visual_history.append(dict(frame_id=3,visual_time=10.06,box=[10,10,30,30]))
        self.assertIsNone(self.w.at_frame(Frame(2,10.04,self.image)))
        result=self.w.at_frame(Frame(4,10.08,self.image))
        self.assertTrue(result['historical']);self.assertTrue(result['blocked'])
        self.assertEqual(result['frame_id'],3)
        self.w.veto('lost');self.assertIsNone(self.w.at_frame(Frame(4,10.08,self.image)))

    def test_processing_late_does_not_destroy_the_template(self):
        self.w.blocked=False;self.w.semantic_time=10.2
        self.w.observation=dict(visual_time=10.15,box=[10,10,30,30])
        self.now=10.3
        result=self.w.snapshot()
        self.assertEqual(result['reason'],'visual_expired');self.assertTrue(result['visual_active'])
        self.assertEqual(self.w.epoch,0);self.assertIsNotNone(self.w.observation)


class TemplateRefreshTests(unittest.TestCase):
    def setUp(self):
        self.image=np.random.default_rng(42).integers(0,256,(100,200,3),dtype=np.uint8)
        self.detection=dict(box=[10,10,50,30],confidence=.9)
        self.template=TemplateRefresh()
        self.template.applied(Frame(1,10.,self.image),self.detection)

    def test_scale_change_refreshes_only_strong_aligned_confirmed_target(self):
        d=dict(self.detection,box=[10,10,70,40])
        f=Frame(2,11.,self.image)
        self.assertEqual(self.template.reason(f,d,.8,True),'confirmed_scale_change')
        self.assertIsNone(self.template.reason(f,dict(d,confidence=.7),.8,True))
        self.assertIsNone(self.template.reason(f,d,.4,True))
        self.assertIsNone(self.template.reason(f,d,.8,False))
        self.assertIsNone(self.template.reason(Frame(3,10.4,self.image),d,.8,True))

    def test_template_stays_fixed_on_unchanged_view_and_flat_or_blurred_frame(self):
        f=Frame(2,11.,self.image)
        self.assertIsNone(self.template.reason(f,self.detection,.9,True))
        flat=Frame(3,12.,np.full_like(self.image,150))
        self.assertIsNone(self.template.reason(flat,dict(self.detection,box=[10,10,70,40]),.9,True))

    def test_confirmed_new_view_can_replace_template_without_relaxing_loss_gate(self):
        f=Frame(2,11.,255-self.image)
        self.assertEqual(self.template.reason(f,self.detection,.9,True),'confirmed_appearance_change')
        self.template.applied(f,self.detection)
        self.assertIsNone(self.template.reason(Frame(3,12.,255-self.image),self.detection,.9,True))


class NanoTransportTests(unittest.TestCase):
    def test_manual_control_checks_live_veto_while_ack_is_pending(self):
        from follow_neo.transport.manual import ManualControl
        from test_control_scan_fix import AsyncServer
        from test_manual import wait_for
        peer=AsyncServer(True);veto=threading.Event();events=[]
        def provider():
            now=time.monotonic()
            return apply_visual(decision(now),visual(now,blocked=veto.is_set()),now)
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(2)))
        control.update(set());control.enable();control.start()
        wait_for(lambda:control.enabled);control.start_dance()
        # Need a semantic source after Dance was explicitly started.
        deadline=time.monotonic()+1
        while not peer.motion.is_set() and time.monotonic()<deadline:
            control.update(set());time.sleep(.01)
        self.assertTrue(peer.motion.is_set())
        veto.set();control.update(set())
        self.assertTrue(peer.zero.wait(.15))
        self.assertTrue(control.wanted);self.assertEqual(control.mode,'DANCE')
        self.assertFalse(peer.disabled.is_set())
        # The peer can receive neutral before the sending thread appends its
        # diagnostic event; wait on that event rather than racing the logger.
        try:
            wait_for(lambda:any(e=='control_motion_neutralizing' and d['reason']=='visual_authorization_changed' for e,d in events))
        except AssertionError:
            self.fail(repr(dict(events=events,commands=peer.commands,status=control.status)))

    def test_veto_during_ack_sends_neutral_without_releasing_keyboard(self):
        client,peer=socket.socketpair();self.addCleanup(client.close);self.addCleanup(peer.close)
        veto=threading.Event();events=[];received=[]
        channel=ControlChannel(client,lambda keys:VISUAL_CHANGE if keys is not None and veto.is_set() else None,
                               lambda e,d:events.append((e,d)))
        def server():
            peer.settimeout(1.)
            received.append(peer.recv(100));veto.set()
            received.append(peer.recv(100));peer.sendall(b'success\r\nsuccess\r\n')
        thread=threading.Thread(target=server);thread.start()
        self.assertFalse(channel.command('rc .1 0 0 .1',motion_keys=frozenset()))
        thread.join(1)
        self.assertIn(b'rc 0 0 0 0\r\n',b''.join(received))
        self.assertFalse(channel.paused);self.assertFalse(channel.pending)
        self.assertTrue(channel.neutral_acknowledged)
        self.assertTrue(any(e=='control_motion_neutralizing' and d['reason']=='visual_authorization_changed' for e,d in events))

    def test_veto_before_send_never_sends_motion(self):
        client,peer=socket.socketpair();self.addCleanup(client.close);self.addCleanup(peer.close)
        channel=ControlChannel(client,lambda keys:VISUAL_CHANGE if keys is not None else None,lambda *args:None)
        peer.sendall(b'success\r\n')
        self.assertFalse(channel.command('rc 1 0 0 1',motion_keys=frozenset()))
        self.assertEqual(peer.recv(100),b'rc 0 0 0 0\r\n')


if __name__=='__main__':unittest.main()
