"""Offline timing regressions; diagnostics must never grant control authority."""
import json
import socket
import unittest
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.transport.control_transport import ControlChannel,RC_ACK_TIMEOUT
from follow_neo.transport.manual import ManualControl
from follow_neo.capture.video import VideoReceiver
from analysis.verify_recovery_run import verify_rows
from test_dual_class import allowed_decision


class TransportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.client,self.peer=socket.socketpair()
        self.addCleanup(self.client.close);self.addCleanup(self.peer.close)
        self.events=[]
        self.channel=ControlChannel(self.client,lambda keys:None,lambda e,d:self.events.append((e,d)))

    def test_late_fifo_ack_reports_original_command_and_cannot_confirm_neutral(self):
        c=self.channel
        with patch('follow_neo.transport.control_transport.time.monotonic',return_value=10.):
            c._send('rc .1 0 0 0')
        with patch('follow_neo.transport.control_transport.time.monotonic',return_value=10.30):
            c.pause('delayed ACK')
        c.buffer=b'success\r\n'
        with patch('follow_neo.transport.control_transport.time.monotonic',return_value=10.40):c._consume()
        report=c.diagnostics(10.41)
        self.assertTrue(report['last_ack']['late']);self.assertAlmostEqual(report['last_ack']['ack_ms'],400.)
        self.assertEqual(report['last_ack']['command'],'rc .1 0 0 0')
        self.assertFalse(report['last_ack']['neutral']);self.assertEqual(report['late_rc_ack_count'],1)
        self.assertEqual(report['pending_count'],1);self.assertAlmostEqual(report['oldest_pending_age_ms'],110.)
        self.assertFalse(c.neutral_acknowledged);self.assertFalse(c.resume())
        c.buffer=b'success\r\n'
        with patch('follow_neo.transport.control_transport.time.monotonic',return_value=10.45):c._consume()
        self.assertTrue(c.neutral_acknowledged);self.assertTrue(c.paused)
        self.assertEqual(c.diagnostics()['pending_count'],0)
        self.assertTrue(c.diagnostics()['last_ack']['neutral'])
        json.dumps(report,allow_nan=False)

    def test_fast_rc_ack_is_not_late_and_samples_are_bounded(self):
        c=self.channel
        for i in range(125):
            c.outstanding=[(i,'rc 0 0 0 0',float(i),False)];c.buffer=b'success\r\n'
            with patch('follow_neo.transport.control_transport.time.monotonic',return_value=i+.05):c._consume()
        d=c.diagnostics(125.)
        self.assertEqual(d['ack_count'],125);self.assertEqual(d['rc_ack_sample_count'],120)
        self.assertEqual(d['late_rc_ack_count'],0);self.assertAlmostEqual(d['rc_ack_p95_ms'],50.)
        self.assertEqual(d['rc_ack_limit_ms'],RC_ACK_TIMEOUT*1000)
        self.assertFalse(c.paused);self.assertIsNone(d['oldest_pending_age_ms'])

    def test_manual_snapshot_reports_growing_pending_age_without_changing_authority(self):
        c=ManualControl('127.0.0.1')
        c.transport_status=dict(oldest_pending_sent_at=10.,pending_count=1)
        with patch('follow_neo.transport.manual.time.monotonic',return_value=10.5):d=c.control_snapshot()
        self.assertEqual(d['transport']['oldest_pending_age_ms'],500.)
        self.assertFalse(d['active']);self.assertFalse(c.enabled)
        self.assertNotIn('oldest_pending_age_ms',c.transport_status)


class VideoTimingTests(unittest.TestCase):
    def test_decode_gaps_include_sampled_frames_and_do_not_include_initial_wait(self):
        events=[];frames=[]
        v=VideoReceiver('127.0.0.1',processing_fps=10.,on_event=lambda e,d:events.append((e,d)),on_frame=frames.append)
        image=Mock();image.to_ndarray.return_value=np.zeros((8,8,3),np.uint8)
        for t in (10.,10.03,10.07,10.10,10.70):
            with patch('follow_neo.capture.video.time.monotonic',return_value=t+.005):
                v._publish_decoded(image,t);v._publish_pending(t+.005)
        self.assertEqual(v.count,5);self.assertGreater(v.sampled_out,0)
        self.assertEqual(len(frames),v.processed_count)
        d=v.timing_snapshot(10.8)
        self.assertEqual(d['decode_gaps_over_250ms'],1);self.assertEqual(d['decode_gaps_over_500ms'],1)
        self.assertAlmostEqual(d['max_decode_gap_ms'],600.)
        self.assertAlmostEqual(d['decoder_frame_age_ms'],100.)
        self.assertAlmostEqual(d['last_publish_ms'],5.)
        self.assertIsNone(d['bytes_age_ms'])
        self.assertEqual(len(events),1);self.assertEqual(events[0][0],'video_decode_gap')
        self.assertEqual(events[0][1]['decoded_frame_id'],5);json.dumps(d,allow_nan=False)


class SavedRunVerificationTests(unittest.TestCase):
    def rows(self,selected_id=0):
        d=allowed_decision()
        d.update(selected_class_id=selected_id,tracking=dict(logical_target_id=7),frame_size=[640,480],
                 control_status=dict(active=True,mode='DANCE'))
        gap=dict(d,prediction_time=10.1,accepted=False,state='COAST',prediction_recovery=True,
                 measurement_age_ms=120.,selected_class_id=None,selected_class_name=None)
        return [dict(event='control_mode',mode='DANCE',monotonic=9.),
                dict(event='observation',decision=d),dict(event='observation',decision=gap)]

    def test_exact_confirmed_neo_provenance_restores_bounded_gate(self):
        rows=self.rows();before=json.dumps(rows)
        report=verify_rows(rows)
        self.assertEqual(report['eligible_retained_identity_decisions'],1)
        self.assertEqual(report['original_nonzero'],0);self.assertEqual(report['restored_nonzero'],1)
        self.assertEqual(json.dumps(rows),before)

    def test_mini_unconfirmed_reset_or_changed_identity_cannot_supply_provenance(self):
        cases=[self.rows(1)]
        rows=self.rows();rows[1]['decision']['confirmed']=False;cases.append(rows)
        rows=self.rows();rows.insert(2,dict(event='tracking_reset'));cases.append(rows)
        rows=self.rows();rows[2]['decision']['tracking']=dict(logical_target_id=8);cases.append(rows)
        for rows in cases:self.assertEqual(verify_rows(rows)['eligible_retained_identity_decisions'],0)

    def test_stale_decision_remains_blocked_after_identity_restoration(self):
        rows=self.rows();rows[2]['decision']['stale']=True
        report=verify_rows(rows)
        self.assertEqual(report['eligible_retained_identity_decisions'],1)
        self.assertEqual(report['restored_nonzero'],0)


if __name__=='__main__':unittest.main()
