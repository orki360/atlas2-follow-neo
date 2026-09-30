"""Live diagram must never confuse perception, held search and control authority."""
import unittest
from follow_neo.state_diagram import diagram_snapshot,CURRENT,PREVIEW,PAUSED,ERROR


class StateDiagramTests(unittest.TestCase):
    def snapshot(self,state='DIRECTIONAL_SEARCH',phase='SCAN',**control_changes):
        d=dict(state=state,prediction_time=10.,reason='directional_search',spacing_phase='APPROACH',
               edge_search=dict(phase=phase,scan_offset_degrees=20.,scan_target_degrees=80.))
        c=dict(connected=True,active=True,mode='DANCE',time=10.)
        c.update(control_changes)
        return diagram_snapshot(d,c,True,True,10.1)

    def test_active_dance_highlights_policy_and_planner_separately(self):
        v=self.snapshot()
        self.assertEqual((v['state'],v['phase']),('DIRECTIONAL_SEARCH','SCAN'))
        self.assertEqual(v['color'],CURRENT)
        self.assertIn('+20.0',v['angles']);self.assertIn('+80.0',v['angles'])

    def test_paused_control_never_looks_like_executing_search(self):
        v=self.snapshot(paused=True,active=False)
        self.assertEqual(v['phase'],'SCAN');self.assertEqual(v['phase_color'],PAUSED)
        self.assertIn('MOVEMENT PAUSED',v['status'])

    def test_search_pause_and_reacquire_retain_held_planner_phase(self):
        for state in ('SEARCH_PAUSED','REACQUIRE'):
            with self.subTest(state=state):
                v=self.snapshot(state=state,phase='BOOST')
                self.assertEqual(v['phase_color'],PAUSED);self.assertIn('HELD',v['heading'])

    def test_control_off_or_keyboard_override_is_preview(self):
        for changes in (dict(active=False),dict(keyboard_override=True),dict(mode='MANUAL'),dict(connected=False,active=True)):
            with self.subTest(changes=changes):self.assertEqual(self.snapshot(**changes)['color'],PREVIEW)

    def test_stale_decision_clears_old_search_highlight(self):
        d=dict(state='DIRECTIONAL_SEARCH',prediction_time=9.,edge_search=dict(phase='BOOST'))
        v=diagram_snapshot(d,{},True,True,10.)
        self.assertEqual(v['state'],'WAIT_VIDEO');self.assertIsNone(v['phase'])

    def test_disconnect_clears_old_tracking_and_search(self):
        d=dict(state='TRACK',prediction_time=10.,edge_search=dict(phase='SCAN'))
        v=diagram_snapshot(d,{},True,False,10.1)
        self.assertIsNone(v['state']);self.assertIsNone(v['phase'])

    def test_error_wins_over_stale_wait_video(self):
        v=diagram_snapshot(dict(state='TRACK',prediction_time=9.),{},True,True,10.,error='Inference failed')
        self.assertEqual(v['state'],'ERROR');self.assertEqual(v['color'],ERROR)

    def test_search_test_uses_test_phase_and_is_not_dance(self):
        c=dict(connected=True,active=True,mode='SEARCH_TEST',time=10.,
               search_test=dict(phase='BOOST',measured_offset_degrees=15.,target_offset_degrees=80.))
        v=diagram_snapshot(dict(state='TRACK',prediction_time=10.),c,False,True,10.1)
        self.assertIsNone(v['state']);self.assertEqual(v['phase'],'BOOST')
        self.assertIn('DANCE OFF',v['status']);self.assertIn('+15.0',v['angles'])

    def test_old_control_snapshot_cannot_claim_enabled_dance(self):
        self.assertEqual(self.snapshot(time=9.)['color'],PREVIEW)
        self.assertIn('STALE',self.snapshot(time=9.)['status'])

    def test_boost_target_uses_span_boundary_not_unset_scan_target(self):
        d=dict(state='DIRECTIONAL_SEARCH',prediction_time=10.,edge_search=dict(
            phase='BOOST',direction=-1,scan_min_degrees=-80.,scan_max_degrees=0.,scan_target_degrees=0.))
        v=diagram_snapshot(d,{},True,True,10.1)
        self.assertIn('target -80.0',v['angles'])


if __name__=='__main__':unittest.main()
