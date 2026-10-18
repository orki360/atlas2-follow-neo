"""Matching-cost controls and fresh-only publication, no aircraft connections."""
import json
import random
import threading
import time
import unittest
from dataclasses import asdict,replace
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.types import Settings,settings_from_config,Detection,Box
from follow_neo.controller import FollowController
from follow_neo.botsort_motion import BoTSORTTracker
from follow_neo.vendor.botsort.bot_sort import BoTSORT
from follow_neo.cadence import FrameRateGate,LatestFrameSampler
from follow_neo.video import VideoReceiver


def detection(x=200,score=.9,cls=0):
    return Detection(Box(x,180,x+60,220),score,cls,'neo' if cls==0 else 'MINI PRO4')


class MatchingSettingsTests(unittest.TestCase):
    def test_persist_and_migrate_without_changing_old_defaults(self):
        s=settings_from_config({'schema_version':8,'settings':{'confidence':.65}})
        self.assertEqual((s.botsort_match_threshold,s.botsort_low_match_threshold,s.botsort_unconfirmed_match_threshold),(.8,.5,.7))
        changed=replace(s,botsort_match_threshold=.85,botsort_low_match_threshold=.55,botsort_unconfirmed_match_threshold=.75)
        self.assertEqual(settings_from_config({'schema_version':8,'settings':asdict(changed)}),changed)
        for key,value in [('botsort_match_threshold',.99),('botsort_low_match_threshold',.8),
                          ('botsort_unconfirmed_match_threshold',.9),('botsort_match_threshold',float('nan'))]:
            with self.assertRaises(ValueError):replace(s,**{key:value}).validate()

    def tracker(self,threshold):
        t=BoTSORTTracker()
        for i in range(4):t.update([detection()],10+i*.1,10+i*.1,i,.65,.65,match_thresholds=(threshold,.5,.7))
        self.assertTrue(t.confirmed)
        return t

    def test_primary_threshold_changes_real_association_not_confidence(self):
        strict=self.tracker(.8);loose=self.tracker(.85)
        for t in (strict,loose):t.update([detection(240)],10.4,10.4,4,.65,.65)
        a=strict.association_report()['stages'][0]['rows'][0]
        b=loose.association_report()['stages'][0]['rows'][0]
        self.assertEqual(a['outcome'],'cost_above_limit');self.assertEqual(b['outcome'],'matched')
        self.assertAlmostEqual(a['best_cost'],b['best_cost'])
        json.dumps(loose.association_report(),allow_nan=False)

    def test_live_settings_update_keeps_identity_and_logs_all_stage_thresholds(self):
        c=FollowController('botsort',0,'neo');s=Settings(confidence=.65)
        for i in range(4):c.observe([detection()],10+i*.1,10+i*.1,i,(640,480),s)
        identity=c.tracker.logical_target_id
        s=replace(s,botsort_match_threshold=.85,botsort_low_match_threshold=.55,botsort_unconfirmed_match_threshold=.75)
        c.observe([detection()],10.4,10.4,4,(640,480),s)
        self.assertEqual(c.tracker.logical_target_id,identity)
        report=c.tracker.association_report()
        self.assertEqual([v['threshold'] for v in report['stages']],[.85,.55,.75])
        self.assertEqual(report['source_time'],10.4)

    def test_weak_threshold_controls_only_second_stage(self):
        outcomes=[]
        for threshold in (.5,.6):
            t=self.tracker(.8)
            t.update([detection(223,.5)],10.4,10.4,4,.65,.65,match_thresholds=(.8,threshold,.7))
            stages=t.association_report()['stages']
            self.assertEqual(stages[0]['detection_count'],0)
            outcomes.append(stages[1]['rows'][0]['outcome'])
        self.assertEqual(outcomes,['cost_above_limit','matched'])

    def test_tentative_threshold_controls_unconfirmed_engine_track(self):
        outcomes=[]
        for threshold in (.7,.8):
            t=BoTSORTTracker()
            t.update([],9.9,9.9,0,.65,.65,match_thresholds=(.8,.5,threshold))
            t.update([detection()],10.,10.,1,.65,.65)
            t.update([detection(234)],10.1,10.1,2,.65,.65)
            report=t.association_report()
            outcomes.append(report['stages'][2]['rows'][0]['outcome'])
            json.dumps(report,allow_nan=False)
        self.assertEqual(outcomes,['cost_above_limit','matched'])

    def test_relaxed_cost_cannot_acquire_mini_or_low_confidence_or_no_overlap(self):
        s=Settings(confidence=.65,botsort_match_threshold=.9,botsort_low_match_threshold=.7,botsort_unconfirmed_match_threshold=.85)
        for det in (detection(score=.99,cls=1),detection(score=.2)):
            c=FollowController('botsort',0,'neo')
            for i in range(5):c.observe([det],10+i*.1,10+i*.1,i,(640,480),s)
            self.assertFalse(c.tracker.confirmed)
        t=self.tracker(.9);source=t.measurement_time
        t.update([detection(500)],10.4,10.4,4,.65,.65)
        self.assertEqual(t.measurement_time,source)

    def test_report_distinguishes_competition_and_limits_size(self):
        engine=BoTSORT.__new__(BoTSORT);engine.association_diagnostics=[];engine.diagnostic_target_id=99
        tracks=[SimpleNamespace(track_id=i) for i in list(range(19))+[99]]
        engine._associate('primary',np.full((20,1),.2),tracks,[SimpleNamespace(score=.9)],.8)
        report=engine.association_diagnostics[0]
        self.assertEqual(report['match_count'],1);self.assertEqual(len(report['rows']),8)
        self.assertEqual(report['rows'][0]['track_id'],99)
        self.assertTrue(any(r['outcome']=='assignment_competition' for r in report['rows']))
        self.assertEqual(report['rows_omitted'],12)


class SamplingTests(unittest.TestCase):
    def test_retains_early_frame_but_replaces_it_with_newest(self):
        s=LatestFrameSampler();s.offer('a',10.);self.assertEqual(s.take(10.,30),'a')
        s.offer('b',10.01);self.assertIsNone(s.take(10.01,30))
        s.offer('c',10.025);self.assertEqual(s.take(10.034,30),'c')
        self.assertEqual(s.replaced,1);self.assertIsNone(s.take(10.07,30))
        self.assertAlmostEqual(s.last_selected_age_ms,9.)

    def test_stall_expiry_and_reset_never_republish_or_refresh_timestamp(self):
        s=LatestFrameSampler();s.offer(('source',10.),10.)
        self.assertIsNone(s.take(10.101,30));self.assertEqual(s.expired,1)
        s.offer(('source',20.),20.);self.assertEqual(s.take(20.,30),('source',20.))
        self.assertIsNone(s.take(20.01,30))
        s.offer('old_epoch',20.01);s.clear();self.assertIsNone(s.take(20.1,30))
        s.offer('new_epoch',20.1);self.assertEqual(s.take(20.1,30),'new_epoch')

    def test_rate_change_and_slow_source_do_not_invent_frames(self):
        s=LatestFrameSampler();s.offer(1,10.);self.assertEqual(s.take(10.,1),1)
        s.offer(2,10.02);self.assertEqual(s.take(10.02,30),2)
        ids=[]
        for k in range(1000):
            t=20+k*.001
            if k%50==0:s.offer(k,t)
            v=s.take(t,30)
            if v is not None:ids.append(v)
        self.assertEqual(len(ids),20);self.assertEqual(len(set(ids)),20)

    def test_jittered_source_improves_rate_without_unbounded_age(self):
        rng=random.Random(321);arrivals=[];t=10.
        while t<30:arrivals.append(t);t+=rng.gammavariate(.5,.038)
        old=FrameRateGate();before=sum(old.admit(t,30) for t in arrivals)
        s=LatestFrameSampler();j=0;selected=[]
        for k in range(20000):
            now=10+k*.001
            while j<len(arrivals) and arrivals[j]<=now:s.offer(j,arrivals[j]);j+=1
            v=s.take(now,30)
            if v is not None:selected.append((v,now-arrivals[v]))
        self.assertGreater(len(selected),before)
        self.assertEqual(len(set(v for v,age in selected)),len(selected))
        self.assertLessEqual(max(age for v,age in selected),.100)
        self.assertLessEqual(len(selected),601)

    def test_conversion_is_deferred_and_source_time_preserved(self):
        frames=[];r=VideoReceiver('offline.invalid',on_frame=frames.append)
        image=Mock(to_ndarray=Mock(return_value=np.zeros((8,8,3),np.uint8)))
        with patch('follow_neo.video.time.monotonic',return_value=10.02):
            r._publish_decoded(image,10.)
            image.to_ndarray.assert_not_called()
            self.assertTrue(r._publish_pending(10.02))
        self.assertEqual(frames[0].decoded_at,10.);self.assertEqual(r.processed_count,1)
        self.assertEqual(r.timing_snapshot(10.03)['sampling']['capacity'],1)

    def test_invalidation_during_conversion_cannot_publish_old_epoch(self):
        r=VideoReceiver('offline.invalid');image=Mock()
        def convert(**kwargs):
            r._invalidate_stream(ConnectionError('offline'),retry_seconds=.5)
            return np.zeros((8,8,3),np.uint8)
        image.to_ndarray.side_effect=convert
        with patch('follow_neo.video.time.monotonic',return_value=10.01):
            r._publish_decoded(image,10.)
            self.assertFalse(r._publish_pending(10.01))
        self.assertIsNone(r.frames.get());self.assertEqual(r.stream_epoch,1)

    def test_conversion_that_takes_too_long_is_discarded(self):
        r=VideoReceiver('offline.invalid');image=Mock(to_ndarray=Mock(return_value=None))
        r._publish_decoded(image,10.)
        with patch('follow_neo.video.time.monotonic',return_value=10.2):
            self.assertFalse(r._publish_pending(10.01))
        self.assertIsNone(r.frames.get());self.assertEqual(r.processed_count,0)

    def test_publication_worker_delivers_pending_frame_without_next_arrival(self):
        frames=[];event=threading.Event();r=VideoReceiver('offline.invalid')
        def published(frame):frames.append(frame);event.set()
        r.on_frame=published
        r.publisher_thread=threading.Thread(target=r._publish_loop,daemon=True);r.publisher_thread.start()
        try:
            image=Mock(to_ndarray=Mock(return_value=np.zeros((8,8,3),np.uint8)))
            r._publish_decoded(image,time.monotonic());self.assertTrue(event.wait(1))
            event.clear();source=time.monotonic();r._publish_decoded(image,source)
            self.assertTrue(event.wait(1));self.assertEqual(frames[-1].decoded_at,source)
            self.assertEqual([f.frame_id for f in frames],[1,2])
        finally:r.stop()
        self.assertFalse(r.publisher_thread.is_alive())


if __name__=='__main__':unittest.main()
