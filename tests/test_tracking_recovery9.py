"""Time alignment, finite observation sweeps and expiring translation; no aircraft."""
import copy
import math
import unittest
import time
from unittest.mock import Mock
from follow_neo.nano_guard import apply_visual,visual_agreement,transport_visual_reason
from follow_neo.nano_tracker import NanoWorker
from follow_neo.search_sectors import SectorSweep
from follow_neo.recovery_search import RecoverySearch
from follow_neo.pursuit import RecoveryVector,recovery_vector
from follow_neo.dance import dance_command,ZERO
from follow_neo.types import Settings,Kinematics
from follow_neo.centering import DampedTrackingController
import test_tracking_recovery8 as previous


def heading(t,yaw=0):return dict(time=t,yaw_deg=yaw,source='offline')


def aligned_case():
    d,v=previous.IndependentControlTests().inputs()
    for sample,x in zip(d['real_neo_evidence']['samples'],(50,80,110)):
        sample['box']=[x,40,x+30,60]
    v.update(box=[140,40,170,60],visual_time=9.98,frame_id=15)
    v['source_alignment']=dict(box=[110,40,140,60],source_time=9.88,frame_id=12,
        identity=7,generation=1,epoch=2,before_time=9.88,after_time=9.88,
        before_frame_id=12,after_frame_id=12)
    return d,v


class SourceTimeAgreementTests(unittest.TestCase):
    def test_moving_drone_matches_original_frame_and_current_motion(self):
        d,v=aligned_case();original=copy.deepcopy(d)
        self.assertEqual(visual_agreement(d,v,10.)['reason'],'source_aligned_motion')
        out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'yolo_only')
        self.assertGreater(dance_command(out,10.,9.)[0][3],0.)
        self.assertEqual(d,original)

    def test_historical_match_cannot_authorize_drift_to_table(self):
        d,v=aligned_case();v['box']=[140,80,170,100]
        self.assertEqual(visual_agreement(d,v,10.)['reason'],'live_motion_disagreement')
        self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO)

    def test_missing_mismatched_and_old_alignment_are_rejected(self):
        for change in (None,dict(epoch=3),dict(identity=8),dict(generation=2),dict(frame_id=11),
                dict(source_time=9.78),dict(before_time=9.7),dict(after_time=10.1),dict(box=[1,1,5,5]),
                dict(before_frame_id=13),dict(after_frame_id=16)):
            d,v=aligned_case()
            if change is None:v.pop('source_alignment')
            else:v['source_alignment'].update(change)
            self.assertFalse(visual_agreement(d,v,10.)['allowed'],change)
        d,v=aligned_case();v['visual_time']=10.02
        self.assertFalse(visual_agreement(d,v,10.)['allowed'])

    def test_alignment_does_not_bypass_hard_veto_or_deadline(self):
        for reason in ('appearance_or_background_drift','target_texture_lost','target_leaving_frame',
                       'yolo_visual_disagreement','competing_mini_detection'):
            d,v=aligned_case();v.update(blocked=True,reason=reason)
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO,reason)
        d,v=aligned_case()
        self.assertEqual(dance_command(apply_visual(d,v,10.14),10.14,9.)[0],ZERO)

    def test_worker_uses_bracketing_real_updates_and_reset_discards_them(self):
        worker=NanoWorker('.',clock=lambda:10.,mode='guarded',generation=1,factory=Mock())
        worker.identity=7;worker.epoch=2;worker.semantic_time=9.8
        checks=dict(geometry_suspect=False,texture_loss_streak=0,visible_fraction=1.)
        worker.visual_history.extend([dict(visual_time=9.85,frame_id=11,box=[90,40,120,60],checks=checks),
                                      dict(visual_time=9.91,frame_id=13,box=[120,40,150,60],checks=checks)])
        a=worker.snapshot(10.,source_time=9.88,frame_id=12)['source_alignment']
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a['box'][0],105.)
        self.assertIsNone(worker.snapshot(10.,source_time=9.7,frame_id=8)['source_alignment'])
        worker.reset('guarded',2)
        self.assertIsNone(worker.snapshot(10.,source_time=9.88,frame_id=12)['source_alignment'])


class SectorSearchTests(unittest.TestCase):
    def test_directional_circle_finishes_after_six_observation_stops(self):
        for sign in (-1,1):
            sweep=SectorSweep(0,min(0,sign*360),max(0,sign*360),sign)
            self.assertEqual(sweep.targets,[sign*x for x in range(60,361,60)])
            now=10.;fid=0;profile=Mock()
            for target in list(sweep.targets):
                for _ in range(9):
                    now+=.1;fid+=1;sweep.observe(now-.01,fid)
                    sweep.update(target,0.,now,profile)
            self.assertTrue(sweep.done);self.assertEqual(len(sweep.completed),6)

    def test_centered_sweep_covers_both_sides_once(self):
        sweep=SectorSweep(0,-180,180,-1)
        self.assertEqual(sweep.targets,[-60,-120,-180,-120,-60,0,60,120,180])

    def test_old_or_repeated_source_frames_cannot_release_observation_stop(self):
        sweep=SectorSweep(0,0,360,1);profile=Mock()
        for i in range(20):
            t=10+i*.1;sweep.observe(9.9,1);sweep.update(60,0.,t,profile)
        self.assertEqual(sweep.phase,'look');self.assertEqual(sweep.index,0)
        sweep.observe(12.,2);sweep.observe(12.1,3);sweep.update(60,0.,12.2,profile)
        self.assertEqual(sweep.index,1)

    def test_actual_high_yaw_rate_and_suspension_invalidate_look_votes(self):
        sweep=SectorSweep(0,0,360,1);profile=Mock()
        for i in range(12):sweep.update(60,40.,10+i*.1,profile)
        self.assertEqual(sweep.phase,'settle');self.assertIsNone(sweep.look_since)
        sweep.update(60,0.,12,profile);sweep.update(60,0.,12.2,profile)
        sweep.observe(12.3,1);sweep.suspend()
        self.assertEqual(sweep.sources,[]);self.assertIsNone(sweep.look_since)

    def test_production_waits_for_fresh_detector_frames_without_extending_deadline(self):
        r=RecoverySearch();s=Settings(search_yaw_degrees=360)
        r.start_test(10.,s,heading(10.),'RIGHT');r.sectors.require_frames=True
        r.rate.update=Mock(return_value=0.)
        r.update(10.1,True,False,'APPROACH',False,s,heading(10.1,60.))
        for i in range(2,15):
            d=r.update(10+i*.1,True,False,'APPROACH',False,s,heading(10+i*.1,60.))
        self.assertEqual(d['sectors']['phase'],'look')
        self.assertTrue(d['active']);self.assertFalse(d['consumed']);self.assertEqual(d['scan_yaw'],0.)
        until=d['until']
        r.sectors.observe(11.5,101);r.sectors.observe(11.6,102)
        d=r.update(11.7,True,False,'APPROACH',False,s,heading(11.7,60.))
        self.assertEqual(d['sectors']['index'],1);self.assertEqual(d['until'],until)
        d=r.update(until+.01,True,False,'APPROACH',False,s,heading(until+.01,60.))
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'search_timeout')


def vector_case():
    d,v=previous.IndependentControlTests().inputs()
    v['checks'].update(control_suspect=False,appearance_correlation=.8,appearance_unsupported=False)
    d.update(video_time=9.99,heading=heading(10.),state='TRACK',track_box=d['selected_measurement_box'],
             centering_control=dict(lateral_control=dict(reason='confirmed_lateral_motion')))
    d['intent'].update(forward=.09,roll=.045)
    memory=RecoveryVector();memory.remember(d,v,10.,True)
    d.update(recovery_anchor=memory.snapshot(),accepted=False,state='COAST',video_time=10.16,
             prediction_time=10.18,heading=heading(10.18))
    v.update(visual_time=10.16,semantic_time=9.88,deadline=10.26)
    return d,v,memory


class RecoveryVectorTests(unittest.TestCase):
    def test_verified_vector_can_continue_briefly_without_new_identity(self):
        d,v,_=vector_case();self.assertIsNotNone(d['recovery_anchor'])
        out=apply_visual(d,v,10.18)
        self.assertEqual(out['nano_authority'],'recovery_vector')
        values,reason=dance_command(out,10.18,9.)
        self.assertGreater(values[3],0.);self.assertGreater(values[2],0.)
        self.assertEqual(values[1],0.);self.assertIn('Recovery vector',reason)

    def test_fresh_camera_frames_cannot_renew_original_source_deadline(self):
        d,v,_=vector_case();values=[]
        for now in (10.18,10.25,10.35,10.42,10.44):
            d.update(video_time=now-.01,prediction_time=now,heading=heading(now))
            v.update(visual_time=now-.01,deadline=now+.09)
            values.append(recovery_vector(d,v,now)['forward'])
        self.assertEqual(values,sorted(values,reverse=True));self.assertEqual(values[-1],0.)

    def test_heading_rotates_vector_and_large_turn_revokes_it(self):
        d,v,_=vector_case();base=recovery_vector(d,v,10.18)
        d['heading']=heading(10.18,20);turn=recovery_vector(d,v,10.18)
        self.assertLess(turn['roll'],base['roll']);self.assertLessEqual(turn['forward'],base['forward'])
        d['heading']=heading(10.18,36)
        self.assertFalse(recovery_vector(d,v,10.18)['allowed'])

    def test_rotation_accounts_for_unequal_gui_axis_limits(self):
        d,v,_=vector_case();d['heading']=heading(10.18,20.)
        d['recovery_axis_limits']=dict(forward=.05,roll=.08)
        out=recovery_vector(d,v,10.18)
        expected=(.1*.08*math.cos(math.radians(20))-.3*.05*math.sin(math.radians(20)))*out['decay']
        self.assertAlmostEqual(out['roll']*.08,expected)
        d['recovery_axis_limits']['roll']=0.
        self.assertEqual(recovery_vector(d,v,10.18)['roll'],0.)
        d['recovery_axis_limits']['forward']=float('nan')
        self.assertFalse(recovery_vector(d,v,10.18)['allowed'])

    def test_background_class_identity_and_geometry_veto_vector(self):
        for kind in ('appearance_or_background_drift','target_texture_lost','target_leaving_frame',
                     'yolo_visual_disagreement','competing_mini_detection','epoch','geometry','suspect'):
            d,v,_=vector_case()
            if kind=='epoch':v['epoch']+=1
            elif kind=='geometry':v['checks']['geometry_suspect']=True
            elif kind=='suspect':v['checks']['control_suspect']=True
            else:v.update(blocked=True,reason=kind)
            self.assertFalse(recovery_vector(d,v,10.18)['allowed'],kind)

    def test_stale_sensors_close_spacing_and_control_restart_revoke_vector(self):
        for change in (dict(video_time=10.),dict(heading=heading(9.9)),dict(candidate_pending=True),
                dict(spacing_close_guard_armed=True),dict(stale=True),dict(class_support={}),dict(bbox_clipped=True),
                dict(selected_class_id=1),dict(generation=2),dict(state='HOVER_WAIT')):
            d,v,_=vector_case();d.update(change)
            self.assertFalse(recovery_vector(d,v,10.18)['allowed'],change)
        d,v,_=vector_case()
        self.assertFalse(recovery_vector(d,v,10.18,started=9.9)['allowed'])
        d['recovery_anchor']['forward']=float('nan')
        self.assertFalse(recovery_vector(d,v,10.18)['allowed'])

    def test_new_real_stop_and_disabled_control_discard_old_permission(self):
        for enabled in (True,False):
            d,v,m=vector_case();d['accepted']=True;d['intent']['forward']=0.
            m.remember(d,v,10.18,enabled);self.assertIsNone(m.snapshot())

    def test_search_translation_requires_vector_and_stops_during_observation(self):
        d,v,_=vector_case();r=RecoverySearch();s=Settings(search_yaw_degrees=80)
        r.start_test(10.,s,heading(10.),'RIGHT')
        d.update(state='DIRECTIONAL_SEARCH',edge_search=r.update(10.18,True,False,'APPROACH',False,s,heading(10.18)))
        values,_=dance_command(apply_visual(d,v,10.18),10.18,9.)
        self.assertGreater(values[0],0.);self.assertGreater(values[3],0.)
        d['edge_search']['sectors']=dict(phase='look')
        self.assertFalse(recovery_vector(d,v,10.18)['allowed'])
        d.pop('recovery_anchor')
        self.assertEqual(dance_command(apply_visual(d,v,10.18),10.18,9.)[0][1:],(0.,0.,0.))

    def test_pending_transport_rechecks_epoch_and_original_expiry(self):
        d,v,_=vector_case();sent=apply_visual(d,v,10.18)
        bad=dict(v,epoch=3)
        self.assertEqual(transport_visual_reason(sent,dict(sent,nano=bad),10.18),'recovery_vector_revoked')
        self.assertEqual(transport_visual_reason(sent,sent,10.45),'recovery_vector_revoked')

    def test_repeated_ticks_do_not_refresh_memory(self):
        d,v,m=vector_case();before=m.snapshot()
        for t in (10.2,10.3,10.4):m.remember(d,v,t,True)
        self.assertEqual(m.snapshot(),before)

    def test_controller_restart_discards_saved_motion(self):
        from follow_neo.controller import FollowController
        c=FollowController('botsort',0,'neo');_,_,c.recovery_vector=vector_case()
        self.assertIsNotNone(c.recovery_vector.snapshot())
        c.restart_control();self.assertIsNone(c.recovery_vector.snapshot())

    def test_live_controller_seeds_from_real_botsort_target_then_expires(self):
        from follow_neo.controller import FollowController
        from follow_neo.types import Box,Detection
        c=FollowController('botsort',0,'neo');s=Settings()
        for i in range(14):
            t=10+i*.1;now=t+.12
            c.observe([Detection(Box(930,500,1010,530),.95,0,'neo')],t,now,i,(1920,1080),s)
            v=dict(mode='guarded',identity=c.tracker.logical_target_id,generation=1,epoch=1,
                blocked=False,reason='visual_supported',box=[930,500,1010,530],semantic_time=t,
                visual_time=now-.02,deadline=now+.08,frame_id=i*3,checks=dict(geometry_suspect=False,
                    control_suspect=False,texture_loss_streak=0,visible_fraction=1.,appearance_correlation=.8))
            d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
        self.assertIsNotNone(d['recovery_anchor']);self.assertEqual(d['state'],'TRACK')
        c.observe([],t+.1,t+.22,14,(1920,1080),s)
        for delta in (.32,.45,.56):
            now=t+delta;v.update(visual_time=now-.02,deadline=min(t+.4,now+.08))
            d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
            values,_=dance_command(apply_visual(d,v,now),now,9.)
            self.assertEqual(values[3]>0,delta<.55)
            self.assertEqual(d['recovery_anchor']['source_time'],t)


class CombinedMotionTests(unittest.TestCase):
    def compute(self,verified=True,visual=True,rate=50.,age=.18,occluded=False):
        c=DampedTrackingController();c.heading_rate.update=Mock(return_value=rate)
        c.compute(Kinematics(True,150,50,20,10),200,100,age,occluded,10.,Settings(),heading(10.),
            10.-age,.5,current_visual=visual,continuity_verified=verified)
        return c

    def test_combined_forward_uses_sustained_real_history_without_waiting_for_nano(self):
        self.assertGreater(self.compute().forward_factor,0.)
        self.assertEqual(self.compute(verified=False).forward_factor,0.)
        self.assertGreater(self.compute(visual=False).forward_factor,0.)

    def test_fast_spin_and_old_prediction_still_cut_forward(self):
        self.assertEqual(self.compute(rate=70.).forward_factor,0.)
        self.assertEqual(self.compute(age=.3,occluded=True).forward_factor,0.)

    def test_combined_lateral_still_needs_camera_compensated_measurements(self):
        c=DampedTrackingController();c.heading_rate.update=Mock(return_value=20.)
        for i in range(8):
            t=10+i*.1
            observation=dict(time=t-.18,frame_id=i,point=(150+i*6,50),width=200,
                warp=[[1,0,0],[0,1,0]],camera_valid=True)
            cmd,_,_=c.compute(Kinematics(True,150,50,20,10),200,100,.18,False,t,Settings(),
                heading(t),t-.18,.5,lateral_observation=observation,current_visual=True,continuity_verified=True)
        self.assertGreater(cmd.roll,0.)
        observation.update(time=t-.08,frame_id=8,camera_valid=False)
        cmd,_,_=c.compute(Kinematics(True,150,50,20,10),200,100,.18,False,t+.1,Settings(),
            heading(t+.1),t-.08,.5,lateral_observation=observation,current_visual=True,continuity_verified=True)
        self.assertEqual(cmd.roll,0.)


class RecoveryTransportTests(unittest.TestCase):
    def test_search_translation_obeys_gui_caps_and_stops_at_original_expiry(self):
        from follow_neo.manual import ManualControl
        from test_manual import Server,wait_for
        peer=Server();events=[];source=time.monotonic()+.30
        seed,vis,_=vector_case();shift=source-9.88
        seed['recovery_anchor']['source_time']=source
        for sample in seed['real_neo_evidence']['samples']:sample['time']+=shift
        seed['measurement_time']=seed['last_strong_measurement_time']=source
        seed['recovery_anchor']['heading_degrees']=0.
        r=RecoverySearch();s=Settings(search_yaw_degrees=80)
        r.start_test(source+.10,s,heading(source+.10),'RIGHT')
        def provider():
            now=time.monotonic()
            if now<source+.30:return None
            d=copy.deepcopy(seed);v=copy.deepcopy(vis)
            d.update(prediction_time=now,video_time=now-.01,heading=heading(now),state='DIRECTIONAL_SEARCH')
            v.update(semantic_time=source,visual_time=now-.01,deadline=min(source+.4,now+.09))
            d['edge_search']=r.update(now,True,False,'APPROACH',False,s,heading(now))
            return apply_visual(d,v,now)
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.02,vertical=.03,roll=.08,forward=.05))
        control.start_dance();control.update(set());control.enable();control.start()
        while time.monotonic()<source+.8:
            control.update(set());time.sleep(.01)
        sent=[d for e,d in events if e=='flight_command' and d['mode']=='DANCE']
        translating=[d for d in sent if d['values'][3]>0]
        self.assertTrue(translating)
        self.assertTrue(all(d['values'][1]==0 and abs(d['values'][2])<=.08*.20 and d['values'][3]<=.05*.35 for d in translating))
        self.assertTrue(all(d['effective_axis_limits']==dict(yaw=1.,vertical=0.,roll=.08,forward=.05) for d in translating))
        expired=[d for d in sent if d['command_time']>=source+.55]
        self.assertTrue(expired);self.assertTrue(all(d['values'][2:]==[0.,0.] for d in expired))


if __name__=='__main__':unittest.main()
