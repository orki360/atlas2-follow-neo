"""Recorded cadence/ambiguity regressions, with no aircraft connection."""
import unittest
from dataclasses import replace
from follow_neo.perception.botsort_motion import BoTSORTTracker, CANDIDATE_PAUSE_SECONDS
from follow_neo.control.controller import FollowController
from follow_neo.control.dance import SearchSmoother, dance_command
from follow_neo.control.centering import DampedTrackingController
from follow_neo.shared.types import Box, Detection, Settings, Kinematics
from follow_neo.transport.telemetry import heading_retry_seconds
from follow_neo.control.recovery_search import RecoverySearch


def neo(x=200,score=.9):return Detection(Box(x,180,x+60,220),score,0,'neo','original_neo',(.9,.05))
def mini(x=200,scores=(.35,.40)):return Detection(Box(x,180,x+60,220),scores[1],1,'MINI PRO4','minip4',scores)
def heading(t):return dict(time=t,yaw_deg=0.,source='offline_test')


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.tracker=BoTSORTTracker();self.time=10.;self.frame=0
    def feed(self,ds,dt=.078):
        self.time+=dt;self.frame+=1
        return self.tracker.update(ds,self.time,self.time+.125,self.frame,.65,.65)
    def acquire(self):
        for _ in range(5):self.feed([neo()])
        self.assertTrue(self.tracker.confirmed)
    def test_two_strong_frames_survive_recorded_234ms_acquisition_interval(self):
        self.feed([neo()]);self.feed([neo()],.234)
        self.assertTrue(self.tracker.confirmed)
    def test_seven_hits_are_not_erased_by_recorded_187ms_gap(self):
        self.acquire();identity=self.tracker.logical_target_id;self.feed([],4.)
        for _ in range(7):self.assertFalse(self.feed([neo(490)]))
        self.assertEqual(self.tracker.pending['hits'],7)
        self.assertTrue(self.feed([neo(490)],.187))
        self.assertEqual(self.tracker.logical_target_id,identity)
        self.assertEqual(self.tracker.identity_event['observations'],8)
    def test_eight_coherent_frames_at_11hz_with_jitter_fit_finite_window(self):
        self.acquire();self.feed([],4.)
        intervals=(.078,.078,.203,.078,.234,.078,.078,.078)
        for i,dt in enumerate(intervals):
            accepted=self.feed([neo(490)],dt)
            self.assertEqual(accepted,i==7)
        self.assertLess(sum(intervals),CANDIDATE_PAUSE_SECONDS)
    def test_true_gap_and_ambiguous_targets_still_discard_evidence(self):
        self.acquire();self.feed([],4.)
        for _ in range(7):self.feed([neo(490)])
        self.assertFalse(self.feed([neo(490)],.35))
        self.assertEqual(self.tracker.pending['hits'],1)
        stamp=self.tracker.measurement_time
        for _ in range(10):self.feed([neo(490),neo(80)])
        self.assertEqual(self.tracker.measurement_time,stamp)
    def test_pending_budget_expires_without_new_candidate_and_duplicate_cannot_vote(self):
        self.acquire();self.feed([],4.);self.feed([neo(490)])
        t=self.tracker;old=t.pending['hits']
        self.assertFalse(t.update([neo(490)],self.time,self.time+.2,self.frame,.65,.65))
        self.assertEqual(t.pending['hits'],old)
        self.feed([],1.5);self.assertFalse(t.candidate_pending)
    def test_inference_latency_does_not_immediately_expire_candidate_pause(self):
        c=FollowController(tracker_backend='botsort');s=Settings(confidence=.65)
        for i in range(8):
            t=10+i*.078;c.observe([neo()],t,t+.18,i,(640,480),s)
        t+=.078;c.observe([neo(255)],t,t+.18,9,(640,480),s)
        d=c.tick(t+.19,640,480,t+.18,s,heading(t+.19))
        self.assertTrue(d['tracking']['candidate_pending']);self.assertTrue(d['candidate_pending'])
        self.assertFalse(d['edge_search']['active']);self.assertFalse(any(d['intent'].values()))
        d=c.tick(t+.51,640,480,t+.5,s,heading(t+.51))
        self.assertFalse(d['candidate_pending'])
    def test_search_handoff_keeps_strong_evidence_at_recorded_234ms_interval(self):
        r=RecoverySearch();s=Settings();r.start_test(10.,s,heading(10.),'LEFT')
        for i,t in enumerate((10.10,10.334,10.568)):
            r.observe(True,False,True,neo().box,Kinematics(),640,480,t,i,s)
            result=r.update(t+.125,False,False,'APPROACH',False,s,heading(t+.125))
            if i<2:
                self.assertTrue(result['verifying']);self.assertFalse(result['active'])
                self.assertEqual(result['verify_hits'],i+1)
        self.assertEqual(result['phase'],'IDLE');self.assertFalse(result['verifying'])


class ClassMemoryTests(unittest.TestCase):
    def setUp(self):
        self.c=FollowController('botsort',0,'neo');self.s=Settings(confidence=.65)
        self.t=10.;self.i=0
    def feed(self,ds,dt=.078):
        self.t+=dt;self.i+=1
        self.c.observe(ds,self.t,self.t+.12,self.i,(640,480),self.s)
        return self.c.tick(self.t+.125,640,480,self.t+.12,self.s,heading(self.t+.125))
    def acquire(self):
        for _ in range(10):d=self.feed([neo()])
        self.assertTrue(d['confirmed']);return d
    def test_near_tie_bridges_only_recent_proven_neo_without_forward_or_new_strong_vote(self):
        before=self.acquire();identity=before['target_identity']['logical_target_id']
        strong=before['last_strong_measurement_time']
        d=self.feed([mini()])
        self.assertTrue(d['accepted']);self.assertIsNotNone(d['class_support'])
        self.assertEqual(d['last_strong_measurement_time'],strong)
        self.assertEqual(d['target_identity']['logical_target_id'],identity)
        self.assertEqual(d['raw_detections'][0]['class_name'],'MINI PRO4')
        self.assertEqual(d['target_class_source'],'temporal_identity')
        values,_=dance_command(d,self.t+.125,9.)
        self.assertEqual(values[2:],(0.,0.));self.assertLessEqual(abs(values[0]),.25)
        self.assertEqual(d['intent']['forward'],0.)
        d=self.feed([neo()]);self.assertIsNone(d['class_support'])
        self.assertEqual(d['target_class_source'],'accepted_measurement')
    def test_ambiguous_mini_cannot_acquire_from_nothing(self):
        for _ in range(12):d=self.feed([mini()])
        self.assertFalse(d['confirmed']);self.assertIsNone(d['measurement_time'])
    def test_clear_mini_wrong_geometry_and_competing_boxes_cannot_refresh_target(self):
        for detections in ([mini(scores=(.02,.99))],[mini(400)],[mini(),mini(205)]):
            self.setUp();before=self.acquire();d=self.feed(detections)
            self.assertFalse(d['accepted']);self.assertIsNone(d['class_support'])
            self.assertEqual(d['measurement_time'],before['measurement_time'])
    def test_ambiguous_frames_do_not_renew_lease_or_allow_motion_after_expiry(self):
        self.acquire();strong=self.c.tracker.last_strong_time
        for _ in range(9):d=self.feed([mini()])
        self.assertEqual(d['last_strong_measurement_time'],strong)
        self.assertLessEqual(d['measurement_time'],strong+.30)
        self.assertEqual(d['intent']['forward'],0.);self.assertEqual(d['intent']['roll'],0.)
        # Expired ambiguous centering cannot authorize TRACK, even if a cached
        # decision still says accepted. A search based on earlier real NEO is separate.
        expired=dict(d,state='TRACK',accepted=True)
        self.assertFalse(any(dance_command(expired,self.t+.125,9.)[0]))
    def test_expired_class_bridge_does_not_permanently_block_real_neo_loss_search(self):
        before=self.acquire();real=before['measurement_time'];turns=[]
        for _ in range(22):
            d=self.feed([mini()]);values,_=dance_command(d,self.t+.125,9.)
            self.assertEqual(values[2:],(0.,0.))
            if d['state']=='DIRECTIONAL_SEARCH' and any(values):
                turns.append(values);self.assertLessEqual(d['edge_search']['source_time'],real)
        self.assertTrue(turns)
    def test_enable_and_reset_require_real_neo_again(self):
        self.acquire();started=self.t+.01;self.c.restart_control()
        d=self.feed([mini()])
        self.assertFalse(any(dance_command(d,self.t+.125,started)[0]))
        self.c.reset();d=self.feed([mini()]);self.assertFalse(d['confirmed'])
    def test_minis_winning_overlap_do_not_build_class_memory(self):
        for _ in range(10):d=self.feed([neo(),mini(scores=(.02,.99))])
        self.assertTrue(d['confirmed'])  # explicit NEO behavior remains available
        d=self.feed([mini()]);self.assertIsNone(d['class_support'])


class MotionTimingTests(unittest.TestCase):
    def test_search_resume_and_reverse_ramp_but_stop_and_braking_are_immediate(self):
        ramp=SearchSmoother();last=0.
        for i in range(25):
            values=ramp.update((-1.,0.,0.,0.),10+i*.05,True)
            self.assertLessEqual(abs(values[0]-last),.050001);last=values[0]
        self.assertEqual(last,-1.)
        self.assertEqual(ramp.update((0.,0.,0.,0.),11.3,False),(0.,0.,0.,0.))
        self.assertLessEqual(abs(ramp.update((-1.,0.,0.,0.),11.362,True)[0]),.062001)
        self.assertEqual(ramp.update((1.,0.,0.,0.),11.4,True)[0],0.)
        self.assertGreater(ramp.update((1.,0.,0.,0.),11.45,True)[0],0.)
        self.assertEqual(ramp.update((.01,0.,0.,0.),11.46,True)[0],.01)
    def test_short_gap_keeps_alignment_history_without_increasing_it(self):
        c=DampedTrackingController();k=Kinematics(True,320,240,40,30);s=Settings()
        c.compute(k,640,480,0,False,10,s);before=c.forward_factor
        c.compute(k,640,480,.15,True,10.15,s)
        self.assertEqual(c.forward_factor,before)
        c.compute(k,640,480,.3,True,10.3,s);self.assertEqual(c.forward_factor,0.)
    def test_heading_reconnect_backoff_is_fast_initially_and_bounded(self):
        self.assertEqual([heading_retry_seconds(i) for i in (1,2,3,4,100)],[.2,.4,.8,1.,1.])


if __name__=='__main__':unittest.main()
