"""Fast crossing recovery, visible motion envelope and bounded combined search."""
import copy
import unittest
from dataclasses import asdict, replace
from unittest.mock import Mock
import numpy as np
from follow_neo.shared.types import Settings, Box, Detection, Kinematics, settings_from_config
from follow_neo.control.recovery_search import RecoverySearch, scan_command
from follow_neo.control.search_sectors import SectorSweep
from follow_neo.control.centering import DampedTrackingController
from follow_neo.presentation.overlay import draw_forward_window
from follow_neo.control.nano_guard import apply_visual, transport_visual_reason
from follow_neo.control.pursuit import recovery_vector
from follow_neo.control.dance import dance_command
from follow_neo.perception.nano_tracker import NanoWorker
from test_tracking_recovery9 import vector_case, heading


class SearchDirectionTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(search_yaw_degrees=360)
        self.r=RecoverySearch();self.r.start_test(10.,self.s,heading(10.),'CENTER')
        self.r.rate.update=Mock(return_value=60.)
        self.until=self.r.until

    def feed(self,source,cx,fid):
        ds=[] if cx is None else [Detection(Box(cx-40,480,cx+40,520),.94,0,'neo')]
        self.r.observe_candidate(ds,source,fid,(1920,1080),self.s)
        return self.r.update(source+.15,True,False,'APPROACH',False,self.s,heading(source+.15),candidate_pending=bool(ds))

    def test_fast_crossing_changes_route_before_braking_finishes(self):
        for i,cx in enumerate((1687,1628,1574,965,322)):
            d=self.feed(10.2+i*.1,cx,i)
        self.assertGreaterEqual(d['candidate_cue']['hits'],5)
        self.assertTrue(d['candidate_guidance'])
        self.assertEqual(d['scan_yaw'],0.)  # still braking the old right turn
        self.assertLess(d['scan_target_degrees'],0.)
        self.r.rate.update=Mock(return_value=0.)
        d=self.feed(10.7,None,5)
        self.assertTrue(d['direction_memory']);self.assertLess(d['scan_yaw'],0.)
        decision=dict(edge_search=d,state='DIRECTIONAL_SEARCH',prediction_time=10.85,
                      accepted=False,stale=False,spacing_phase='APPROACH')
        self.assertLess(scan_command(decision,10.85,9.),0.)
        self.assertEqual(d['until'],self.until)
        d=self.r.update(11.5,True,False,'APPROACH',False,self.s,heading(11.5))
        self.assertFalse(d['direction_memory'])
        self.assertLess(d['scan_target_degrees'],0.)  # no return to old route
        self.assertEqual(d['until'],self.until)

    def test_mirrored_crossing_and_mini_conflict(self):
        self.r.first_scan_direction=-1;self.r.target=-60
        self.r.rate.update=Mock(return_value=-50.)
        for i,cx in enumerate((365,646,1005,1166,1538,1749)):
            d=self.feed(10.2+i*.1,cx,i)
        self.assertGreater(d['scan_target_degrees'],0.)
        self.assertEqual(d['scan_yaw'],0.)
        neo=Detection(Box(1700,480,1780,520),.94,0,'neo')
        self.r.observe_candidate([neo,Detection(neo.box,.9,1,'mini')],10.8,8,(1920,1080),self.s)
        self.assertIsNone(self.r.direction_hint)

    def test_one_or_repeated_detection_never_grants_guidance(self):
        d=self.feed(10.2,300,1)
        self.assertTrue(d['paused']);self.assertFalse(d['candidate_guidance'])
        d=self.feed(10.2,300,1)
        self.assertFalse(d['direction_memory']);self.assertFalse(d['candidate_guidance'])

    def test_old_velocity_cannot_override_latest_left_reversal(self):
        r=RecoverySearch()
        for i,cx in enumerate((673,679,695,712,731,753,706)):
            t=10+i*.09
            r.observe(True,False,True,Box(cx-40,480,cx+40,520),
                      Kinematics(True,cx,500,80,40,-350,0),1920,1080,t,i,self.s)
        self.assertEqual(r.estimated_direction,-1)
        self.assertEqual(r.direction_source,'latest_side_after_reversal')

    def test_memorized_direction_never_renews_search_deadline(self):
        self.feed(10.2,1500,1);self.feed(10.3,1600,2)
        d=self.r.update(self.until+.01,True,False,'APPROACH',False,self.s,heading(self.until+.01))
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'search_timeout')


class EnvelopeTests(unittest.TestCase):
    def compute(self,ex=.5,ey=0.,rate=30.,settings=None,visual=False):
        c=DampedTrackingController();c.heading_rate.update=Mock(return_value=rate)
        k=Kinematics(True,100+100*ex,50+50*ey,20,10)
        c.compute(k,200,100,.12,False,10.,settings or Settings(),heading(10.),9.88,ex,
                  current_visual=visual,continuity_verified=True)
        return c

    def test_width_height_soft_edges_and_independent_real_yolo(self):
        self.assertEqual(self.compute().forward_factor,1.)
        self.assertEqual(self.compute(ex=.81).forward_factor,0.)
        self.assertEqual(self.compute(ey=.71).forward_factor,0.)
        middle=self.compute(ex=.72).forward_factor
        self.assertGreater(middle,0.);self.assertLess(middle,1.)
        s=replace(Settings(),forward_window_width=.4,forward_window_height=.3)
        self.assertEqual(self.compute(settings=s).forward_factor,0.)
        self.assertEqual(self.compute(ex=0,ey=.4,settings=s).forward_factor,0.)

    def test_yaw_limits_and_optional_nano_requirement(self):
        self.assertGreater(self.compute(rate=50.).forward_factor,0.)
        self.assertEqual(self.compute(rate=50.,settings=replace(Settings(),forward_yaw_rate_limit=45.)).forward_factor,0.)
        self.assertEqual(self.compute(settings=replace(Settings(),forward_yaw_demand_limit=.2,coordinated_turns=False)).forward_factor,0.)
        self.assertEqual(self.compute(settings=replace(Settings(),forward_require_nano=True),rate=50.).forward_factor,0.)
        self.assertGreater(self.compute(settings=replace(Settings(),forward_require_nano=True),visual=True).forward_factor,0.)

    def test_settings_roundtrip_and_hard_limits(self):
        s=replace(Settings(),forward_window_width=.65,forward_window_height=.35,search_look_seconds=.2)
        self.assertEqual(settings_from_config(dict(schema_version=8,settings=asdict(s))),s)
        for key,value in (('forward_window_width',1.),('forward_window_height',.9),
                ('forward_yaw_rate_limit',100.),('nano_visual_max_age_seconds',.4),
                ('search_translation_seconds',3.),('search_forward_fraction',1.),('search_side_fraction',1.),
                ('search_look_seconds',0.),('forward_require_nano','false')):
            with self.assertRaises(ValueError):replace(s,**{key:value}).validate()

    def test_four_corners_use_frame_proportions_without_filling(self):
        image=np.zeros((1000,1600,3),np.uint8)
        m=draw_forward_window(image,dict(motion_settings=dict(forward_window_width=.6,forward_window_height=.4)))
        self.assertEqual(m['display_box'],[320,300,1280,700])
        for x,y in ((320,300),(1280,300),(320,700),(1280,700)):
            self.assertTrue(np.all(image[y,x]==160))
        self.assertFalse(image[500,800].any())
        self.assertLess(np.count_nonzero(image),5000)


def search_case(now=10.58):
    d,v,_=vector_case();options=asdict(Settings())
    d['motion_settings']=options
    d['recovery_anchor'].update(search_seconds=1.,search_side=1,search_forward_fraction=.35,search_side_fraction=.20)
    r=RecoverySearch();s=Settings(search_yaw_degrees=90)
    r.start_test(10.3,s,heading(10.3),'RIGHT')
    d.update(state='DIRECTIONAL_SEARCH',prediction_time=now,video_time=now-.01,heading=heading(now),
        edge_search=r.update(now,True,False,'APPROACH',False,s,heading(now)))
    v.update(visual_time=now-.01,deadline=now+.09)
    return d,v


class CombinedSearchTests(unittest.TestCase):
    def test_real_controller_loss_reaches_combined_search_and_expires(self):
        from follow_neo.control.controller import FollowController
        c=FollowController('botsort',0,'neo');s=Settings(search_yaw_degrees=360)
        for i in range(14):
            source=10+i*.1;now=source+.12
            box=[1300,480,1380,520]
            c.observe([Detection(Box(*box),.95,0,'neo')],source,now,i,(1920,1080),s)
            v=dict(mode='guarded',identity=c.tracker.logical_target_id,generation=1,epoch=1,
                blocked=False,reason='visual_supported',box=box,semantic_time=source,
                visual_time=now-.01,deadline=now+.09,frame_id=i*3,checks=dict(geometry_suspect=False,
                    control_suspect=False,texture_loss_streak=0,visible_fraction=1.,appearance_correlation=.8))
            d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
        self.assertIsNotNone(d['recovery_anchor'])
        last_source=d['recovery_anchor']['source_time'];motions=[]
        for i in range(1,14):
            now=source+.12+i*.1
            c.observe([],now-.12,now,13+i,(1920,1080),s,allow_search_cues=True)
            v.update(visual_time=now-.01,deadline=now+.09)
            d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
            out=apply_visual(d,v,now);values,_=dance_command(out,now,9.)
            if d['state']=='DIRECTIONAL_SEARCH':motions.append((now,values))
        self.assertTrue(any(v[0]>0 and v[2]>0 and v[3]>0 for _,v in motions),motions)
        self.assertTrue(all(v[2:]==(0.,0.) for t,v in motions if t>=last_source+1.))

    def test_turn_forward_and_side_after_normal_grace(self):
        d,v=search_case();out=apply_visual(d,v,10.58)
        values,_=dance_command(out,10.58,9.)
        self.assertGreater(values[0],0.);self.assertGreater(values[2],0.);self.assertGreater(values[3],0.)
        self.assertEqual(values[1],0.)
        self.assertLessEqual(values[2],.20);self.assertLessEqual(values[3],.35)
        self.assertEqual(out['recovery_motion']['reason'],'measured_bearing_search')
        self.assertIsNone(transport_visual_reason(out,out,10.58))

    def test_short_budget_keeps_same_side_policy_and_zero_axis_caps(self):
        d,v=search_case(10.4)
        d['edge_search'].update(scan_yaw=.1,active=True)
        d['motion_settings']['search_translation_seconds']=.55
        d['recovery_anchor']['search_seconds']=.55
        out=recovery_vector(d,v,10.4)
        self.assertTrue(out['allowed']);self.assertEqual(out['reason'],'measured_bearing_search')
        self.assertGreater(out['roll'],0.)
        d['recovery_axis_limits']=dict(forward=0.,roll=0.)
        out=recovery_vector(d,v,10.4)
        self.assertEqual((out['forward'],out['roll']),(0.,0.))

    def test_source_expiry_close_distance_sensor_age_and_ambiguity_stop_translation(self):
        for change in (dict(spacing_close_guard_armed=True),dict(video_time=10.),
                       dict(heading=heading(10.)),dict(candidate_pending=True),dict(class_support={}),
                       dict(selected_class_id=1),dict(generation=2),dict(bbox_clipped=True)):
            d,v=search_case();d.update(change)
            self.assertFalse(recovery_vector(d,v,10.58)['allowed'],change)
        d,v=search_case(10.9)
        self.assertEqual(recovery_vector(d,v,10.9)['reason'],'vector_expired')
        self.assertEqual(dance_command(apply_visual(d,v,10.9),10.9,9.)[0][2:],(0.,0.))

    def test_observation_reversal_and_candidate_cannot_use_old_translation(self):
        for change in (dict(sectors=dict(phase='look')),dict(scan_yaw=-.2),
                       dict(candidate_guidance=True),dict(direction_memory=True),dict(active=False)):
            d,v=search_case();d['edge_search'].update(change)
            self.assertFalse(recovery_vector(d,v,10.58)['allowed'],change)

    def test_background_and_epoch_veto_remain_hard(self):
        for reason in ('appearance_or_background_drift','appearance_unverified','competing_mini_detection',
                       'yolo_visual_disagreement','target_texture_lost','target_leaving_frame'):
            d,v=search_case();v.update(blocked=True,reason=reason)
            self.assertFalse(recovery_vector(d,v,10.58)['allowed'],reason)
        d,v=search_case();v['epoch']+=1
        self.assertFalse(recovery_vector(d,v,10.58)['allowed'])

    def departure(self):
        d,v=search_case();a=d['recovery_anchor'];checks=dict(v['checks'])
        checks.update(appearance_unsupported=False,visible_fraction=.5)
        v.update(blocked=True,reason='target_leaving_frame',epoch=a['epoch']+1,
            exit_context=dict(prior_epoch=a['epoch'],identity=a['identity'],generation=a['generation'],
                time=10.2,box=[195,40,225,60],frame_size=[200,100],checks=checks))
        d['bbox_clipped']=True
        return d,v

    def test_verified_edge_departure_uses_original_budget_never_nano_box(self):
        d,v=self.departure();out=recovery_vector(d,v,10.58)
        self.assertTrue(out['allowed']);self.assertTrue(out['measured_exit'])
        self.assertEqual(out['until'],9.88+1.)
        for key,value in (('control_suspect',True),('geometry_suspect',True),
                          ('texture_loss_streak',1),('appearance_correlation',.1),('appearance_unsupported',True)):
            bad=copy.deepcopy(v);bad['exit_context']['checks'][key]=value
            self.assertFalse(recovery_vector(d,bad,10.58)['allowed'],key)
        for key,value in (('box',[50,40,80,60]),('prior_epoch',999),('time',10.5),('identity',999)):
            bad=copy.deepcopy(v);bad['exit_context'][key]=value
            self.assertFalse(recovery_vector(d,bad,10.58)['allowed'],key)
        self.assertFalse(recovery_vector(d,dict(v,epoch=v['epoch']+1),10.58)['allowed'])

    def test_transport_revokes_new_background_even_with_old_exit_context(self):
        d,v=self.departure();out=apply_visual(d,v,10.58)
        latest=apply_visual(d,dict(v,reason='appearance_or_background_drift'),10.58)
        self.assertEqual(transport_visual_reason(out,latest,10.58),'recovery_vector_revoked')

    def test_nano_veto_keeps_context_only_until_next_reset(self):
        w=NanoWorker('.',clock=lambda:10.,mode='guarded',generation=1,factory=Mock())
        w.identity=7;w.semantic_time=9.88;w.blocked=False
        w.veto('target_leaving_frame',exit_context=dict(time=9.99))
        self.assertEqual(w.snapshot()['exit_context']['prior_epoch'],0)
        self.assertTrue(w.snapshot()['blocked'])
        w.reset('guarded',2)
        self.assertIsNone(w.snapshot()['exit_context'])


class AdaptiveLookTests(unittest.TestCase):
    def test_two_new_frames_release_look_before_old_350_ms_minimum(self):
        s=SectorSweep(0,0,360,1);p=Mock()
        s.update(60,0.,10.,p);s.update(60,0.,10.09,p)
        s.observe(10.10,1);s.observe(10.20,2)
        s.update(60,0.,10.26,p)
        self.assertEqual(s.index,1)

    def test_shorter_time_never_substitutes_for_new_source_frames(self):
        s=SectorSweep(0,0,360,1);p=Mock()
        for t in (10.,10.1,10.4,10.6):
            s.observe(9.9,1);s.update(60,0.,t,p)
        self.assertEqual(s.index,0)


if __name__=='__main__':unittest.main()
