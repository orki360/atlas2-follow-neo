"""Regression for 20261015_174513: isolated state/evidence tests, no aircraft."""
import copy
import unittest
from follow_neo.types import Settings,Box,Detection,Kinematics
from follow_neo.recovery_search import RecoverySearch
from follow_neo.nano_guard import RealNeoEvidence,sustained_real_neo,apply_visual,transport_visual_reason,current_visual_support
from follow_neo.dance import dance_command,ZERO
from follow_neo.controller import FollowController
from test_nano_bridge import decision,visual
from test_tracking_recovery7 import heading
import test_tracking_recovery7 as recovery7
from follow_neo.approach import LateralEvidence


class RecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(search_yaw_degrees=360)
        self.r=RecoverySearch()
        self.r.observe(True,False,True,Box(900,500,1040,540),Kinematics(),1920,1080,10.,1,self.s)

    def cue(self,t,fid,x):
        self.r.observe_candidate([Detection(Box(x,500,x+150,540),.95,0,'neo')],t,fid,(1920,1080),self.s)
        return self.r.update(t+.12,True,False,'APPROACH',False,self.s,heading(t+.12),candidate_pending=True)

    def test_recorded_partial_reacquisition_does_not_consume_recovery(self):
        self.cue(10.3,2,600);cue=self.cue(10.4,3,450)
        self.assertEqual(cue['candidate_scope'],'recovery_overlay')
        self.assertFalse(cue['candidate_only']);self.assertTrue(cue['candidate_guidance'])
        self.assertEqual(cue['first_scan_direction'],-1)
        deadline=cue['until'];self.assertEqual(cue['angle_degrees'],360)
        self.r.observe(True,False,True,Box(250,500,400,540),Kinematics(),1920,1080,10.5,4,self.s)
        d=self.r.update(10.62,False,False,'APPROACH',False,self.s,heading(10.62))
        self.assertTrue(d['verifying']);self.assertFalse(d['consumed']);self.assertEqual(d['phase'],'SCAN')
        self.r.observe_candidate([],10.7,5,(1920,1080),self.s)
        d=self.r.update(11.,True,False,'APPROACH',False,self.s,heading(11.))
        self.assertTrue(d['active']);self.assertFalse(d['consumed']);self.assertLess(d['scan_yaw'],0.)
        self.assertEqual(d['until'],deadline)
        d=self.r.update(deadline+.01,True,False,'APPROACH',False,self.s,heading(deadline+.01))
        self.assertFalse(d['active']);self.assertTrue(d['consumed'])

    def test_cue_expiry_without_accepted_frame_keeps_original_scan_budget(self):
        self.cue(10.3,2,600);d=self.cue(10.4,3,450);until=d['until']
        self.r.observe_candidate([],10.5,4,(1920,1080),self.s)
        d=self.r.update(10.62,True,False,'APPROACH',False,self.s,heading(10.62))
        self.assertFalse(d['candidate_guidance']);self.assertEqual(d['phase'],'SCAN')
        self.assertFalse(d['consumed']);self.assertEqual(d['until'],until)

    def test_already_consumed_search_only_gets_bounded_look_and_verification(self):
        self.r.end('search_timeout');self.cue(12.,2,600);d=self.cue(12.1,3,450)
        self.assertTrue(d['candidate_only']);self.assertEqual(d['angle_degrees'],30.)
        until=d['until']
        self.r.observe(True,False,True,Box(250,500,400,540),Kinematics(),1920,1080,12.2,4,self.s)
        d=self.r.update(12.32,False,False,'APPROACH',False,self.s,heading(12.32))
        self.assertTrue(d['verifying']);self.assertFalse(d['consumed'])
        self.r.observe_candidate([],12.4,5,(1920,1080),self.s)
        d=self.r.update(12.7,True,False,'APPROACH',False,self.s,heading(12.7))
        self.assertFalse(d['active']);self.assertTrue(d['consumed']);self.assertEqual(d['until'],until)

    def test_full_reacquisition_clears_old_cue_contract(self):
        self.cue(10.3,2,600);self.cue(10.4,3,450)
        for i in range(3):
            t=10.5+i*.1
            self.r.observe(True,False,True,Box(250,500,400,540),Kinematics(),1920,1080,t,4+i,self.s)
            d=self.r.update(t+.12,False,False,'APPROACH',False,self.s,heading(t+.12))
        self.assertEqual(d['phase'],'IDLE');self.assertFalse(d['consumed'])
        self.assertEqual(d['candidate_scope'],'none')

    def test_video_loss_and_operator_stop_still_revoke_search(self):
        for flag in ('stale','control'):
            self.setUp();self.cue(10.3,2,600);self.cue(10.4,3,450)
            d=self.r.update(10.6,True,flag=='stale','APPROACH',False,self.s,heading(10.6),control_available=flag!='control')
            self.assertFalse(d['active']);self.assertTrue(d['consumed'])
            self.assertFalse(self.cue(10.7,4,250)['active'])


class RealEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.h=RealNeoEvidence();self.det=Detection(Box(110,40,140,60),.94,0,'neo')

    def feed(self,t,i,ds=None,**changes):
        args=dict(accepted=True,confirmed=True,box=self.det.box,identity=7,
                  source=t,frame_id=i,size=(200,100),threshold=.65)
        args.update(changes);self.h.observe([self.det] if ds is None else ds,**args)

    def sample_decision(self):
        return decision(generation=1,measurement_id=12,real_neo_evidence=self.h.snapshot())

    def test_unique_source_frames_required_and_duplicate_ticks_do_not_vote(self):
        self.feed(9.68,10)
        for _ in range(8):self.feed(9.68,10)
        self.assertFalse(sustained_real_neo(self.sample_decision()))
        self.feed(9.78,11);self.feed(9.88,12)
        self.assertTrue(sustained_real_neo(self.sample_decision()))
        d=self.sample_decision();d['measurement_time']=9.98
        self.assertFalse(sustained_real_neo(d))

    def test_missing_weak_mini_and_multiple_neo_break_history(self):
        for ds in ([],[Detection(self.det.box,.7,0,'neo')],
                   [self.det,Detection(self.det.box,.8,1,'MINI PRO4')],
                   [self.det,Detection(Box(10,10,30,30),.8,0,'neo')]):
            self.setUp();self.feed(9.68,10);self.feed(9.78,11);self.feed(9.88,12,ds)
            self.assertEqual(self.h.snapshot()['samples'],[])

    def test_identity_jump_gap_and_reversal_of_time_do_not_reuse_votes(self):
        for changes in (dict(identity=8),dict(source=10.2),dict(source=9.65,frame_id=9),
                        dict(box=Box(1,1,10,10)),dict(accepted=False),dict(confirmed=False)):
            self.setUp();self.feed(9.68,10);self.feed(9.78,11);self.feed(9.88,12,**changes)
            self.assertFalse(sustained_real_neo(self.sample_decision()))

    def test_controller_exports_real_evidence_and_control_restart_discards_it(self):
        c=FollowController('botsort',0,'neo');s=Settings(confidence=.65)
        for i in range(12):c.observe([self.det],10+i*.1,10.12+i*.1,i,(200,100),s)
        d=c.tick(11.22,200,100,11.2,s,heading(11.22))
        self.assertTrue(sustained_real_neo(d))
        c.restart_control();self.assertEqual(c.real_neo_evidence.snapshot()['samples'],[])


class IndependentControlTests(unittest.TestCase):
    def inputs(self):
        d,v=recovery7.SoftNanoEvidenceTests().inputs()
        h=RealNeoEvidence();det=Detection(Box(*d['selected_measurement_box']),.94,0,'neo')
        for i,t in enumerate((9.68,9.78,9.88)):
            h.observe([det],True,True,det.box,7,t,10+i,(200,100),.65)
        d['real_neo_evidence']=h.snapshot();d['measurement_id']=12
        return d,v

    def test_sustained_real_neo_survives_soft_pixel_ambiguity_only(self):
        for change in (dict(background_like_motion=True),dict(box_flow_error_px=15.)):
            d,v=self.inputs();v['checks'].update(change)
            out=apply_visual(d,v,10.)
            self.assertEqual(out['nano_authority'],'yolo_only')
            self.assertEqual(out['nano_handoff']['yolo_evidence'],'sustained')
            self.assertGreater(dance_command(out,10.,9.)[0][3],0.)
            d['real_neo_evidence']['samples'].pop(0)
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO)

    def test_hard_conflict_and_unreliable_identity_still_win(self):
        for kind in ('background','appearance_or_background_drift','target_texture_lost',
                     'competing_mini_detection','target_leaving_frame','yolo_visual_disagreement',
                     'geometry','texture','visibility','box','class','generation','identity','stale'):
            d,v=self.inputs();v['checks']['background_like_motion']=True
            if kind=='geometry':v['checks']['geometry_suspect']=True
            elif kind=='texture':v['checks']['texture_loss_streak']=1
            elif kind=='visibility':v['checks']['visible_fraction']=.8
            elif kind=='box':v['box']=[1,1,5,5]
            elif kind=='class':d['raw_detections'].append(dict(d['raw_detections'][0],class_id=1))
            elif kind=='generation':v['generation']=8
            elif kind=='identity':v['identity']=8
            elif kind=='stale':d['stale']=True
            else:v.update(blocked=True,reason=kind)
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO,kind)

    def test_history_does_not_extend_source_deadline_or_grant_nano_only_motion(self):
        for now,change in ((10.131,{}),(10.,dict(accepted=False,state='COAST')),
                           (10.,dict(class_support={'history':True})),(10.,dict(last_strong_measurement_time=9.78))):
            d,v=self.inputs();d.update(change);v['checks']['background_like_motion']=True
            self.assertEqual(dance_command(apply_visual(d,v,now),now,9.)[0],ZERO)

    def test_fresh_yolo_is_not_demoted_while_nano_checks_latest_anchor(self):
        d,v=self.inputs();v.update(semantic_time=9.78)
        v['checks'].update(appearance_correlation=.7,control_suspect=False,appearance_unsupported=False)
        original=copy.deepcopy(d);out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'yolo_only')
        self.assertEqual(out['intent'],d['intent']);self.assertEqual(d,original)
        self.assertEqual(out['nano_handoff']['forward_reason'],'fresh_yolo_pending_nano_anchor')
        self.assertGreater(dance_command(out,10.,9.)[0][3],0.)
        for reason in ('competing_mini_detection','target_leaving_frame','yolo_visual_disagreement'):
            latest=apply_visual(d,dict(v,blocked=True,reason=reason),10.01)
            self.assertEqual(transport_visual_reason(out,latest,10.01),'visual_control_blocked')

    def test_current_stop_intent_cannot_be_replaced_with_old_forward(self):
        d,v=self.inputs();v['semantic_time']=9.78
        v['checks'].update(appearance_correlation=.7,control_suspect=False,appearance_unsupported=False)
        d['intent']['forward']=0.
        self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0][3],0.)


class FreshVisualLateralTests(unittest.TestCase):
    def inputs(self):
        d,v=IndependentControlTests().inputs()
        d.update(measurement_time=9.82,last_strong_measurement_time=9.82,measurement_age_ms=180.,
            raw_detection_time=9.82,track_box=dict(x1=110.,y1=40.,x2=140.,y2=60.),
            centering_control=dict(lateral_control=dict(uses_visual_support=True)))
        v.update(semantic_time=9.82)
        v['checks'].update(appearance_correlation=.7,control_suspect=False,appearance_unsupported=False)
        return d,v

    def test_delayed_source_needs_fresh_independent_visual_support(self):
        for supported in (False,True):
            e=LateralEvidence();value=0
            for i in range(8):
                t=10.+i*.1
                o=dict(time=t,frame_id=i,point=(320+i*6,240),width=640,
                       warp=[[1,0,0],[0,1,0]],camera_valid=True)
                value=e.update(o,t+.18,True,'',current_visual=supported)
            self.assertEqual(value>0,supported)
            self.assertEqual(e.update(o,t+.251,True,'',current_visual=True),0.)

    def test_camera_pan_still_cannot_create_lateral_motion(self):
        e=LateralEvidence()
        for i in range(12):
            t=10+i*.1;o=dict(time=t,frame_id=i,point=(320+i*6,240),width=640,
                            warp=[[1,0,6],[0,1,0]],camera_valid=True)
            self.assertEqual(e.update(o,t+.18,True,'',current_visual=True),0.)

    def test_visual_proof_checks_age_identity_quality_and_geometry(self):
        d,v=self.inputs();box=d['track_box']
        self.assertTrue(current_visual_support(v,10.,7,1,box))
        for change in (dict(visual_time=9.899),dict(semantic_time=9.5),dict(identity=8),dict(generation=2),
                       dict(box=[1,1,5,5]),dict(blocked=True,reason='background'),dict(mode='shadow'),
                       dict(checks=dict(v['checks'],geometry_suspect=True)),
                       dict(checks=dict(v['checks'],appearance_correlation=.1))):
            self.assertFalse(current_visual_support(dict(v,**change),10.,7,1,box),change)

    def test_send_gate_and_reapplication_revoke_lateral_on_visual_expiry(self):
        d,v=self.inputs();out=apply_visual(d,v,10.)
        self.assertGreater(dance_command(out,10.,9.)[0][2],0.)
        expired=dict(v,visual_time=9.899)
        latest=apply_visual(d,expired,10.)
        self.assertEqual(latest['intent']['roll'],0.)
        self.assertGreater(latest['intent']['forward'],0.)
        self.assertEqual(dance_command(dict(out,nano=expired,nano_authority='yolo_only'),10.,9.)[0][2],0.)
        self.assertEqual(transport_visual_reason(out,latest,10.),'visual_lateral_expired')
        self.assertEqual(apply_visual(out,dict(v,mode='off'),10.)['intent']['roll'],0.)

    def test_new_source_and_current_pixels_cannot_authorize_lateral_during_reacquisition(self):
        d,v=self.inputs();d['state']='REACQUIRE'
        self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0][2],0.)


if __name__=='__main__':unittest.main()
