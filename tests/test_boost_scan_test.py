"""Diagnostic and Dance share continuous handoff, braking and stop handling."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from follow_neo.search_test import SearchTest
from follow_neo.types import Settings
from follow_neo.manual import ManualControl
from follow_neo.session import FollowSession
from follow_neo.state_diagram import diagram_snapshot, PAUSED, CURRENT
from test_manual import Server, wait_for
from test_search_debug import heading


class BoostScanTests(unittest.TestCase):
    def make(self, direction='RIGHT'):
        events=[]
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,direction,
                        lambda e,d:events.append((e,d)),boost=True)
        return test,events

    def test_both_directions_handoff_without_a_fixed_zero_pause(self):
        for direction,sign in [('LEFT',-1),('RIGHT',1)]:
            with self.subTest(direction=direction):
                test,events=self.make(direction)
                test.search.rate.update=Mock(return_value=0.)
                self.assertEqual(test.update(heading(10.1,sign),10.1),(sign,0,0,0))
                self.assertEqual(test.last_sample['target_offset_degrees'],sign*80)
                self.assertEqual(test.update(heading(10.36,sign*8),10.36),(sign,0,0,0))
                self.assertEqual(test.last_sample['phase'],'BOOST')
                values=test.update(heading(10.6,sign*35),10.6)
                self.assertEqual(test.last_sample['phase'],'SCAN')
                self.assertGreater(values[0]*sign,0);self.assertLessEqual(abs(values[0]),1)
                self.assertEqual(test.last_sample['boost_exit_reason'],'boundary_braking')
                self.assertFalse(test.last_sample['transition_pause'])
                self.assertGreater(test.update(heading(10.7,sign*35),10.7)[0]*sign,0)
                self.assertFalse(test.last_sample['transition_pause'])
                transitions=[d for e,d in events if e=='search_test_transition']
                self.assertEqual([(d['from_phase'],d['to_phase']) for d in transitions],
                                 [('IDLE','BOOST'),('BOOST','SCAN')])

    def test_boundary_braking_still_exits_before_timer(self):
        test,_=self.make();test.search.rate.update=Mock(return_value=90.)
        self.assertEqual(test.update(heading(10.1,70),10.1),(0,0,0,0))
        self.assertEqual(test.last_sample['boost_exit_reason'],'boundary_braking')
        self.assertFalse(test.last_sample['transition_pause'])  # zero comes from braking

    def test_actual_reversal_keeps_pause_in_both_directions(self):
        for direction,sign in [('LEFT',-1),('RIGHT',1)]:
            test,_=self.make(direction);test.search.rate.update=Mock(return_value=0.)
            self.assertEqual(test.update(heading(10.1,sign*80),10.1),(0,0,0,0))
            self.assertTrue(test.last_sample['transition_pause'])
            self.assertEqual(test.last_sample['target_offset_degrees'],0.)
            self.assertEqual(test.update(heading(10.24,sign*80),10.24),(0,0,0,0))
            self.assertGreater(test.update(heading(10.26,sign*80),10.26)[0]*-sign,0)

    def test_center_is_rejected_and_existing_test_still_starts_scan(self):
        with self.assertRaises(ValueError):self.make('CENTER')
        test=SearchTest(Settings(),heading(10),10,'CENTER',lambda *a:None)
        self.assertEqual(test.search.phase,'SCAN');self.assertEqual(test.kind,'SCAN_ONLY')
        self.assertEqual(test.update(heading(10.1),10.1),(0,0,0,0))
        self.assertTrue(test.last_sample['transition_pause'])

    def test_stale_heading_and_operator_stop_suppress_boost(self):
        test,events=self.make()
        self.assertEqual(test.update(heading(9),10.1),(0,0,0,0))
        test.stop(10.12,'operator_stop')
        self.assertEqual(test.update(heading(10.15,2),10.15),(0,0,0,0))
        self.assertEqual(test.last_sample['phase'],'DONE')
        self.assertTrue(test.last_sample['tail'])
        self.assertEqual([d['to_phase'] for e,d in events if e=='search_test_transition'],['BOOST','DONE'])

    def test_one_measured_cycle_stops_after_boost_and_scan(self):
        test,_=self.make()
        for t,yaw in [(10.1,2),(10.36,8),(10.6,80),(10.8,0)]:
            values=test.update(heading(t,yaw),t)
        self.assertEqual(values,(0,0,0,0));self.assertEqual(test.reason,'scan_complete')
        self.assertEqual(test.last_sample['phase'],'DONE')

    def test_diagram_shows_continuous_handoff_and_later_reversal_pause(self):
        test,_=self.make();test.update(heading(10.1,1),10.1)
        test.search.rate.update=Mock(return_value=0.)
        test.update(heading(10.36,35),10.36)
        control=dict(connected=True,active=True,time=10.36,mode='SEARCH_TEST',search_test=test.last_sample)
        view=diagram_snapshot(None,control,False,True,10.36)
        self.assertIsNone(view['state']);self.assertEqual(view['phase'],'SCAN')
        self.assertEqual(view['phase_color'],CURRENT)
        self.assertEqual(view['label'],'BOOST -> SCAN TEST')
        for text in ('Requested YAW','rate estimate','boundary_braking','search_test_'):
            self.assertIn(text,view['test_details'])
        self.assertNotIn('SCAN pause',view['test_details'])
        test.update(heading(10.5,80),10.5)
        control.update(time=10.5,search_test=test.last_sample)
        view=diagram_snapshot(None,control,False,True,10.5)
        self.assertEqual(view['phase_color'],PAUSED);self.assertIn('SCAN pause',view['test_details'])
        test.stop(10.6,'operator_stop');test.update(heading(10.6,80),10.6)
        control.update(mode='MANUAL',time=10.6,search_test=test.last_sample)
        self.assertEqual(diagram_snapshot(None,control,False,True,10.6)['phase'],'DONE')
        control['connected']=False
        self.assertIsNone(diagram_snapshot(None,control,False,False,10.6)['phase'])

    def test_dedicated_log_includes_transitions_samples_commands_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=FollowSession.__new__(FollowSession);session.log=Mock();session.output=Path(tmp)
            session.flight_commands_sent=0;session.search_test_log=None
            test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'LEFT',session.record_control,boost=True)
            test.update(heading(10.1,-1),10.1);test.update(heading(10.36,-35),10.36)
            session.record_control('control_command_attempt',dict(command='rc 0 0 0 0',sequence=1))
            session.record_control('control_command_ack',dict(command='rc 0 0 0 0',sequence=1,reply='success'))
            test.stop(10.4,'operator_stop');test.finish(10.5)
            rows=[json.loads(x) for x in (Path(tmp)/('search_test_'+test.run_id)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['test_kind'],'BOOST_SCAN')
            self.assertIn('application_revision',rows[0])
            scan_sample=next(r for r in rows if r['event']=='search_test_sample' and r['phase']=='SCAN')
            self.assertFalse(scan_sample['transition_pause']);self.assertLess(scan_sample['requested_yaw'],0)
            self.assertEqual(rows[0]['application_revision'],'straight-approach-13')
            self.assertEqual([r['to_phase'] for r in rows if r['event']=='search_test_transition'],['BOOST','SCAN','DONE'])
            self.assertEqual(rows[-1]['event'],'search_test_summary')
            self.assertFalse(rows[-1]['tail_complete'])
            self.assertEqual(rows[-1]['boost_exit_reason'],'boundary_braking')
            self.assertIsNone(session.search_test_log)

    def test_control_worker_sends_continuous_sequence_and_stop(self):
        peer=Server();events=[]
        c=ManualControl('127.0.0.1',peer.connect,on_event=lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.set_axis_limits(dict(yaw=.01,vertical=1.,roll=1.,forward=1.))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        origin=time.monotonic()
        def moving_heading():
            now=time.monotonic();return heading(now,-min(35.,(now-origin)*70))
        c.start_search_test(Settings(search_yaw_degrees=80),moving_heading,'LEFT',boost=True)
        def resumed_scan():
            c.update(set())
            commands=[d for e,d in events if e=='flight_command' and d['mode']=='SEARCH_TEST']
            return any(d['edge_search']['phase']=='SCAN' and d['values'][0]<0 for d in commands)
        wait_for(resumed_scan)
        commands=[d for e,d in events if e=='flight_command' and d['mode']=='SEARCH_TEST']
        self.assertTrue(any(d['edge_search']['phase']=='BOOST' and d['values'][0]==-1 for d in commands))
        first_scan=next(d for d in commands if d['edge_search']['phase']=='SCAN')
        self.assertLess(first_scan['values'][0],0)
        self.assertTrue(all(d['values'][1:]==[0,0,0] for d in commands))
        c.stop_search_test();index=len(peer.commands)
        wait_for(lambda:(c.update(set()) or 'rc 0.0000 0.0000 0.0000 0.0000' in peer.commands[index:]))
        self.assertEqual(c.mode,'MANUAL')


if __name__=='__main__':unittest.main()
