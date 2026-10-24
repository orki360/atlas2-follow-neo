"""Bounded HARD recovery on offline sockets; no phone/aircraft or live model."""
import multiprocessing as mp
import threading
import time
import unittest
from unittest.mock import patch

from follow_neo.diagnostics import activity
from follow_neo.ipc.mailbox import InputState, Mailbox, FrameSlot
from follow_neo.runtime.worker import Runtime
from follow_neo.transport.manual import ManualControl, HARD_TCP_WAIT_SECONDS
from follow_neo.transport.control_transport import RC_ACK_TIMEOUT, UNEXPECTED_REPLY
from test_control_scan_fix import AsyncServer
from test_dance import decision
import test_hard_runtime as hard_helpers
from test_manual import wait_for

TIMEOUT=f'Control acknowledgement delayed beyond {RC_ACK_TIMEOUT:.2f}s'


class HardWaitTests(unittest.TestCase):
    def setUp(self):
        activity.set_enabled(True)
        self.addCleanup(activity.set_enabled,True)

    def paused_control(self):
        c=ManualControl('offline.invalid');c.enabled=c.wanted=c.connected=True
        c.mode='DANCE';c.updated=10.;c.set_hard_wait(True)
        with patch('follow_neo.transport.manual.time.monotonic',return_value=10.):
            c.pause_control(TIMEOUT,automatic_recovery=True)
        return c

    def test_fixed_ack_limit_and_original_pause_deadline(self):
        self.assertEqual(RC_ACK_TIMEOUT,.25);self.assertEqual(HARD_TCP_WAIT_SECONDS,1.)
        c=self.paused_control()
        # Diagnostic snapshots are deliberately stale/disabled in HARD.
        c.transport_status={'ambiguous':True,'oldest_pending_age_ms':100000}
        for now in (10.1,10.4,10.999):
            with patch('follow_neo.transport.manual.time.monotonic',return_value=now):
                c.updated=now;c.pause_control(TIMEOUT,automatic_recovery=True)
                self.assertIsNone(c.enforce_hard_wait())
                self.assertEqual(c.pause_started,10.)
        with patch('follow_neo.transport.manual.time.monotonic',return_value=11.):
            self.assertIn('exceeded',c.enforce_hard_wait())
        self.assertFalse(c.wanted);self.assertFalse(c.auto_resume_allowed)
        c.communication_paused=False
        self.assertIsNotNone(c.enforce_hard_wait())  # latched until leaving HARD
        c.set_hard_wait(False);self.assertIsNone(c.enforce_hard_wait())
        for invalid in (None,float('nan'),float('inf'),10.2):
            c=self.paused_control();c.pause_started=invalid;c.updated=10.1
            with patch('follow_neo.transport.manual.time.monotonic',return_value=10.1):
                self.assertIsNotNone(c.enforce_hard_wait())

    def test_only_unambiguous_ack_delay_is_eligible(self):
        for reason in ('UI heartbeat lost','Window focus lost',UNEXPECTED_REPLY,
                       'Control command rejected: denied','Control send timed out', 'local processing error'):
            c=self.paused_control()
            with patch('follow_neo.transport.manual.time.monotonic',return_value=10.1):
                c.updated=10.1;c.pause_control(reason,automatic_recovery=True)
                self.assertIsNotNone(c.enforce_hard_wait(),reason)
                self.assertFalse(c.wanted)
        for change in ('hold_before_pause','input_keys','auto_resume_allowed','heartbeat'):
            c=self.paused_control()
            if change=='heartbeat':c.updated=9.
            elif change=='auto_resume_allowed':c.auto_resume_allowed=False
            elif change=='input_keys':c.input_keys={'up'}
            else:setattr(c,change,True)
            with patch('follow_neo.transport.manual.time.monotonic',return_value=10.1):
                self.assertIsNotNone(c.enforce_hard_wait(),change)

    def test_normal_mode_pause_policy_is_unchanged(self):
        c=self.paused_control();c.set_hard_wait(False)
        with patch('follow_neo.transport.manual.time.monotonic',return_value=15.):
            self.assertIsNone(c.enforce_hard_wait())
        self.assertTrue(c.wanted);self.assertTrue(c.auto_resume_allowed)

    def live_hard(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        r=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,1000),FrameSlot(ctx,1000))
        r.session=hard_helpers.HardTests.session(self);r.session_id=r.identity=1
        peer=AsyncServer(True)
        # Independent clocks: fresh decisions do not automatically refresh
        # the real measurement or the camera source after a recovery.
        source={'measurement':None,'strong':None,'video':None}
        def current():
            now=time.monotonic()
            measured=now if source['measurement'] is None else source['measurement']
            strong=now if source['strong'] is None else source['strong']
            video=now if source['video'] is None else source['video']
            return decision(now,measurement_time=measured,last_strong_measurement_time=strong,
                            video_time=video,continuous_dance=True)
        r.manual=ManualControl('127.0.0.1',peer.connect,on_event=r.log_control,decision_provider=current)
        r.session.control_provider=r.control
        stop=threading.Event()
        def pulse():
            while not stop.wait(.01):inputs.pulse(set(),1,origin=r.input_origin)
        heartbeat=threading.Thread(target=pulse,daemon=True)
        worker=threading.Thread(target=r.input_loop,daemon=True)
        def cleanup():
            self.flush(peer)
            r.manual.release()
            r.exit_hard();stop.set();r.close()
            heartbeat.join(1);worker.join(1)
        self.addCleanup(cleanup)
        r.manual.start();inputs.pulse(set(),1);r.consume_input();r.manual.enable()
        heartbeat.start();worker.start();wait_for(lambda:r.manual.enabled)
        hard_helpers.HardTests.enter(self,r)
        self.assertTrue(peer.motion.wait(1.))
        wait_for(lambda:r.manual.communication_paused)
        wait_for(lambda:len(peer.withheld)>=2)
        return r,peer,source,stop

    def flush(self,peer,count=None):
        try:
            if count is None:peer.block=False;count=len(peer.withheld)
            if count:
                del peer.withheld[:count]
                peer.server.sendall(b'success\r\n'*count)
        except OSError:pass

    def motions(self,peer):
        return [(t,c) for t,c in peer.commands if c.startswith('rc ') and any(float(v) for v in c.split()[1:])]

    def test_wait_requires_both_acks_then_new_real_measurement_and_video_without_io(self):
        r,p,source,_=self.live_hard();c=r.manual
        before=r.session.log.path.read_bytes();samples=dict(r.session.scheduling.values)
        self.assertFalse(activity.enabled);self.assertTrue(r.hard)
        self.assertFalse(r.inputs.hard_abort.is_set());self.assertTrue(c.preserve_tracking)
        self.assertEqual(len(self.motions(p)),1);self.assertFalse(p.disabled.is_set())
        source['measurement']=source['strong']=source['video']=time.monotonic()
        self.flush(p,1)  # original motion ACK, NOT the neutral ACK
        time.sleep(.07)
        self.assertTrue(c.communication_paused);self.assertEqual(len(self.motions(p)),1)
        self.flush(p)
        wait_for(lambda:not c.communication_paused)
        self.assertTrue(c.enabled);self.assertEqual(c.mode,'DANCE')
        self.assertIsNotNone(c.resume_measurement_after)
        # One second is a maximum wait, never a fixed recovery sleep.
        self.assertLess(c.resume_measurement_after-c.pause_started,.8)
        time.sleep(.08);self.assertEqual(len(self.motions(p)),1)
        source['measurement']=source['video']=None  # strong NEO confirmation is still old
        time.sleep(.08);self.assertEqual(len(self.motions(p)),1)
        source['video']=c.resume_measurement_after-.01;source['strong']=None
        time.sleep(.08);self.assertEqual(len(self.motions(p)),1)
        source['video']=None
        wait_for(lambda:len(self.motions(p))>1)
        self.assertFalse(r.inputs.hard_abort.is_set());self.assertTrue(r.hard)
        self.assertEqual(sum(cmd=='enable' for _,cmd in p.commands),1)
        self.assertFalse(p.disabled.is_set())
        self.assertEqual(r.session.log.path.read_bytes(),before)
        self.assertEqual(r.session.scheduling.values,samples)
        self.assertIsNone(r.session.runtime_sample)
        self.assertEqual(r.session.log.queue.qsize(),0)

    def test_missing_ack_exits_after_fixed_wait_and_late_reply_cannot_resume(self):
        r,p,_,_=self.live_hard();beg=r.manual.pause_started
        wait_for(r.inputs.hard_abort.is_set)
        self.assertGreaterEqual(time.monotonic()-beg,HARD_TCP_WAIT_SECONDS)
        self.assertIn('exceeded',r.manual.hard_wait_error)
        self.assertFalse(r.manual.wanted)
        wait_for(p.disabled.is_set)
        self.flush(p);time.sleep(.08)
        self.assertEqual(len(self.motions(p)),1)
        self.assertEqual(sum(cmd=='enable' for _,cmd in p.commands),1)

    def test_resume_gate_enforces_expiry_even_without_worker_poll(self):
        r,p,_,_=self.live_hard()
        # Keep input alive, but suspend the worker's guard: only the transport
        # resume path can prevent an already-expired wait from resuming here.
        with patch.object(r.manual,'enforce_hard_wait',return_value=None):
            with r.manual.lock:r.manual.pause_started=time.monotonic()-1.01
            self.flush(p)
            wait_for(lambda:r.manual.hard_wait_error is not None)
            self.assertFalse(r.manual.wanted)
            self.assertTrue(r.manual.communication_paused)
            self.assertEqual(len(self.motions(p)),1)

    def test_release_and_heartbeat_loss_during_wait_never_resume(self):
        for fault in ('release','heartbeat','video','rejected','ambiguous'):
            with self.subTest(fault=fault):
                r,p,_,pulse_stop=self.live_hard()
                if fault=='release':r.inputs.release()
                elif fault=='heartbeat':pulse_stop.set()
                elif fault=='video':r.session.receiver.state='ERROR'
                elif fault=='rejected':
                    p.withheld.pop(0);p.server.sendall(b'denied\r\n')
                else:
                    # Deliver the excess reply in the SAME read as the known
                    # ACKs; do not race a separate write against valid resume.
                    count=len(p.withheld);p.withheld.clear()
                    p.server.sendall(b'success\r\n'*(count+1))
                wait_for(r.inputs.hard_abort.is_set)
                self.assertFalse(r.manual.wanted)
                self.flush(p);time.sleep(.05)
                self.assertEqual(len(self.motions(p)),1)
                self.assertEqual(sum(cmd=='enable' for _,cmd in p.commands),1)
                r.exit_hard();r.close();pulse_stop.set()
