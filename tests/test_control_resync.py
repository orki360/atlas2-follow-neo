"""No aircraft connections: reproduce extra ACKs and the inspected Android parser."""
import socket
import threading
import time
import unittest
from unittest.mock import patch
from follow_neo.transport.control_transport import ControlChannel,UNEXPECTED_REPLY,ZERO
from follow_neo.transport.manual import ManualControl
from test_dance import decision
from test_manual import wait_for


class MarkerPeer:
    def __init__(self,clock,variant='ok'):
        self.clock=clock;self.variant=variant;self.wire=[];self.chunks=[]
    def setsockopt(self,*a):pass
    def settimeout(self,t):self.timeout=t
    def sendall(self,data):
        self.wire.append(data)
        if data.startswith(b'python_sync_'):
            echo=b'Unknown command: '+data
            self.chunks={
                'ok':[b'success\r\nsucc',b'ess\r\n'+echo[:20],echo[20:]],
                'generic':[b'success\r\n']*3,
                'wrong':[b'Unknown command: old_marker\r\n'],
                'rejected':[b'denied\r\n'+echo],
                'trailing':[echo+b'success\r\n'],
                'missing':[],
            }[self.variant][:]
    def recv(self,n):
        self.clock[0]+=self.timeout
        if self.chunks:return self.chunks.pop(0)
        raise socket.timeout()


class NeutralSynchronizationTests(unittest.TestCase):
    def case(self,variant='ok',allowed=lambda:True):
        now=[10.];peer=MarkerPeer(now,variant);events=[]
        with patch('follow_neo.transport.control_transport.time.monotonic',side_effect=lambda:now[0]):
            c=ControlChannel(peer,lambda keys:None,lambda e,d:events.append((e,d)))
            c._unmatched(b'success\r\n')
            ok=c.synchronize_neutral(allowed)
        return c,peer,events,ok

    def test_only_fresh_unique_echo_confirms_neutral_with_fragmented_old_replies(self):
        c,p,events,ok=self.case()
        self.assertTrue(ok);self.assertFalse(c.ambiguous);self.assertTrue(c.paused)
        self.assertTrue(c.neutral_acknowledged);self.assertFalse(c.pending)
        self.assertEqual(p.wire[-2],(ZERO+'\r\n').encode())
        self.assertTrue(p.wire[-1].startswith(b'python_sync_'))
        self.assertFalse(any(b'enable' in x or b'disable' in x for x in p.wire))
        self.assertFalse(any(e=='control_command_ack' for e,d in events))
        completed=next(d for e,d in events if e=='control_resync_completed')
        self.assertEqual(completed['quarantined_successes'],2)
        self.assertTrue(c.resume())

    def test_generic_ack_wrong_marker_rejection_and_trailing_data_never_resume(self):
        for variant in ('generic','wrong','rejected','trailing','missing'):
            with self.subTest(variant=variant):
                c,p,events,ok=self.case(variant)
                self.assertFalse(ok);self.assertTrue(c.ambiguous)
                self.assertFalse(c.neutral_acknowledged);self.assertFalse(c.resume())
                self.assertTrue(any(e=='control_resync_failed' for e,d in events))

    def test_repeated_fault_gets_no_second_automatic_attempt(self):
        c,p,events,ok=self.case();self.assertTrue(c.resume())
        c._unmatched(b'success\r\n');sent=len(p.wire)
        self.assertFalse(c.can_resync)
        self.assertFalse(c.synchronize_neutral(lambda:True));self.assertEqual(len(p.wire),sent)

    def test_non_success_partial_rejection_and_send_timeout_are_not_eligible(self):
        c,p,events,ok=self.case()
        for raw in (b'succ',b'error\r\n',b'success\r\ndenied\r\n',b'\r\n',b'success\r\n'*1000):
            c.resync_attempted=False;c._unmatched(raw)
            self.assertFalse(c.can_resync)
        c.unexpected_success_only=True;c.pause_reason='Control send timed out'
        self.assertFalse(c.can_resync)
        c.pause_reason=UNEXPECTED_REPLY;c.rejected=True
        self.assertFalse(c.can_resync)

    def test_cancelled_operator_or_heartbeat_never_confirms_or_resumes(self):
        calls=[0]
        def allowed():calls[0]+=1;return calls[0]<3
        c,p,events,ok=self.case(allowed=allowed)
        self.assertFalse(ok);self.assertTrue(c.paused);self.assertFalse(c.resume())
        self.assertTrue(any(e=='control_resync_failed' for e,d in events))


class AndroidProtocolPeer:
    """Regular RC and unknown-command branches from the read-only Android ZIP."""
    def __init__(self,marker_action=None):
        self.client,self.server=socket.socketpair();self.commands=[]
        self.inject=False;self.marker_action=marker_action;self.connections=0
        threading.Thread(target=self.run,daemon=True).start()
    def connect(self,address,timeout):self.connections+=1;return self.client
    def run(self):
        try:
            with self.server.makefile('rb') as stream:
                for line in stream:
                    cmd=line.decode().strip();self.commands.append(cmd)
                    if cmd.startswith('python_sync_'):
                        if self.marker_action:self.marker_action()
                        reply=('Unknown command: '+cmd+'\r\n').encode()
                    else:
                        reply=b'success\r\n'
                        if self.inject and cmd.startswith('rc '):
                            self.inject=False;reply+=b'success\r\n'
                    self.server.sendall(reply)
        except OSError:pass
        finally:self.server.close()


class ManualSynchronizationTests(unittest.TestCase):
    def run_case(self,mode='DANCE',cancel=None):
        events=[];paused_source=[None];control_ref=[];marker_time=[None]
        def marker():
            marker_time[0]=time.monotonic()
            if cancel=='release':control_ref[0].release()
            if cancel=='focus':control_ref[0].pause_control('Window or settings focus lost')
        peer=AndroidProtocolPeer(marker)
        def provider():
            now=time.monotonic()
            source=paused_source[0] if paused_source[0] is not None else now-.01
            return decision(now,video_time=source,measurement_time=source,measurement_age_ms=(now-source)*1000)
        def event(e,d):
            events.append((e,d))
            if e=='control_resync_completed':paused_source[0]=time.monotonic()-.05
        c=ManualControl('127.0.0.1',peer.connect,event,provider);control_ref.append(c)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        if mode=='DANCE':c.start_dance()
        for _ in range(5):c.update(set());time.sleep(.02)
        peer.inject=True
        for _ in range(20):c.update(set());time.sleep(.02)
        return c,peer,events,paused_source,marker_time

    def test_dance_recovers_same_connection_then_waits_for_post_barrier_measurement(self):
        c,p,events,source,marker=self.run_case()
        self.assertFalse(c.communication_paused);self.assertEqual(p.connections,1)
        self.assertEqual(p.commands.count('enable'),1)
        resumed=[d for e,d in events if e=='control_movement_resumed']
        self.assertEqual(len(resumed),1)
        self.assertTrue(resumed[0]['preserve_tracking']);self.assertTrue(resumed[0]['authority_retained'])
        barrier=resumed[0]['resume_measurement_after']
        after=[d for e,d in events if e=='flight_command' and d.get('command_time',0)>barrier]
        self.assertTrue(after);self.assertTrue(all(not any(d['values']) for d in after))
        source[0]=None
        for _ in range(10):c.update(set());time.sleep(.02)
        self.assertTrue(any(e=='flight_command' and d.get('command_time',0)>barrier and any(d['values']) for e,d in events),
            [(e,d.get('gate_reason',d.get('reason')),d.get('values')) for e,d in events[-15:]])

    def test_manual_never_automatically_recovers_unexpected_ack(self):
        c,p,events,source,marker=self.run_case(mode='MANUAL')
        self.assertTrue(c.communication_paused);self.assertIsNone(marker[0])
        self.assertFalse(any(e=='control_resync_started' for e,d in events))

    def test_operator_release_and_focus_loss_win_over_marker_reply(self):
        for cancel in ('release','focus'):
            with self.subTest(cancel=cancel):
                c,p,events,source,marker=self.run_case(cancel=cancel)
                self.assertTrue(c.communication_paused)
                self.assertFalse(any(e=='control_movement_resumed' for e,d in events))
                self.assertEqual(p.commands.count('enable'),1)
                c.stop();c.thread.join(3)


if __name__=='__main__':unittest.main()
