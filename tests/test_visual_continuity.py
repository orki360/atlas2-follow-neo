"""Behavioral regressions for real detections, bounded bridges and search cues."""
import unittest
import threading
import time
from types import SimpleNamespace
import numpy as np
from follow_neo.types import Settings, Box, Detection
from follow_neo.nano_guard import apply_visual, continuity_detection, transport_visual_reason
from follow_neo.nano_tracker import NanoWorker
from follow_neo.video import Frame
from follow_neo.dance import dance_command, ZERO
from follow_neo.recovery_search import RecoverySearch, scan_command
from follow_neo.botsort_motion import BoTSORTTracker
from test_nano_bridge import decision, visual


GOOD=dict(control_suspect=False,geometry_suspect=False,visible_fraction=1.,appearance_correlation=.8)


class ContinuityTests(unittest.TestCase):
    def test_real_yolo_survives_temporarily_unavailable_nano(self):
        out=apply_visual(decision(),visual(blocked=True,reason='waiting_visual_update'),10.)
        self.assertTrue(any(dance_command(out,10.,9.)[0]))
        self.assertEqual(out['nano_authority'],'yolo_only')

    def test_actual_background_veto_still_blocks_even_with_cached_yolo(self):
        for reason in ('appearance_unverified','competing_mini_detection','target_leaving_frame'):
            out=apply_visual(decision(),visual(blocked=True,reason=reason),10.)
            self.assertEqual(dance_command(out,10.,9.)[0],ZERO)

    def test_independent_yolo_command_is_revoked_when_model_generation_changes(self):
        sent=apply_visual(decision(),visual(blocked=True,reason='waiting_visual_update'),10.)
        latest=dict(sent,nano=dict(sent['nano'],generation=sent['nano']['generation']+1))
        self.assertEqual(transport_visual_reason(sent,latest,10.),'visual_generation_changed')

    def bridge(self,now):
        anchor=dict(time=9.72,identity=7,forward=.1,allowed=True,box=[90,40,110,60])
        d=decision(now,measurement_time=9.72,measurement_age_ms=(now-9.72)*1000,
            last_strong_measurement_time=9.72,accepted=False,state='COAST',
            verified_forward_anchor=anchor)
        v=visual(now,semantic_time=9.72,deadline=min(10.12,now+.08),
                 box=[90,40,110,60],checks=dict(GOOD))
        return d,v

    def test_forward_bridge_decays_without_new_yolo_and_expires(self):
        values=[]
        for now in (10.,10.03,10.06,10.08):
            d,v=self.bridge(now);out=apply_visual(d,v,now)
            values.append(dance_command(out,now,9.)[0][3])
            self.assertEqual(out['measurement_time'],9.72)
            self.assertEqual(out['measurement_id'],d['measurement_id'])
        self.assertGreater(values[0],0)
        self.assertEqual(values,sorted(values,reverse=True));self.assertEqual(values[-1],0.)
        d,v=self.bridge(10.13)
        self.assertEqual(dance_command(apply_visual(d,v,10.13),10.13,9.)[0],ZERO)

    def test_bridge_cannot_translate_after_enable_or_near_target_or_on_suspect_pixels(self):
        for changes in ({'spacing_close_guard_armed':True},{'bbox_clipped':True},{'spacing_phase':'BRAKE'}):
            d,v=self.bridge(10.);d.update(changes)
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0][3],0.)
        d,v=self.bridge(10.)
        self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.8)[0],ZERO)
        v['checks']=dict(GOOD,appearance_correlation=.05)
        self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0][3],0.)

    def test_refreshing_visual_position_does_not_reset_forward_deadline(self):
        d,v=self.bridge(10.)
        first=apply_visual(d,v,10.)
        later=apply_visual(first,dict(v,visual_time=10.07,deadline=10.12),10.08)
        self.assertEqual(later['measurement_time'],9.72)
        self.assertEqual(dance_command(later,10.08,9.)[0][3],0.)

    def test_bridge_rejects_invalid_forward_command_at_send_time(self):
        d,v=self.bridge(10.);out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'bridge_forward')
        for value in (float('nan'),float('inf'),None,-.1):
            bad=dict(out,intent=dict(out['intent'],forward=value))
            self.assertEqual(dance_command(bad,10.,9.)[0],ZERO)

    def test_pending_forward_is_revoked_at_its_own_deadline_or_new_close_guard(self):
        d,v=self.bridge(10.);sent=apply_visual(d,v,10.)
        latest=dict(sent,nano=dict(sent['nano'],visual_time=10.07,deadline=10.12))
        self.assertEqual(transport_visual_reason(sent,latest,10.071),'visual_forward_expired')
        for changes in ({'spacing_phase':'BRAKE'},{'spacing_close_guard_armed':True},{'bbox_clipped':True}):
            self.assertEqual(transport_visual_reason(sent,dict(latest,**changes),10.01),'visual_forward_revoked')

    def test_source_alignment_and_mini_conflict_gate_real_detection(self):
        frame=Frame(2,10.1,np.zeros((100,200,3),np.uint8))
        neo=dict(box=[90,40,110,60],confidence=.9,class_id=0,class_name='neo')
        result=dict(frame=frame,generation=0,detections=[neo])
        evidence=dict(box=neo['box'],source_time=10.1,frame_id=2,identity=7,generation=0)
        self.assertIsNotNone(continuity_detection(result,evidence,7,Settings()))
        self.assertIsNone(continuity_detection(result,dict(evidence,source_time=10.12),7,Settings()))
        self.assertIsNone(continuity_detection(result,evidence,8,Settings()))
        result['detections'].append(dict(neo,class_id=1,class_name='MINI PRO4',confidence=.7))
        self.assertIsNone(continuity_detection(result,evidence,7,Settings()))
        result['detections']=[dict(neo,class_id=1,class_name='MINI PRO4')]
        self.assertIsNone(continuity_detection(result,evidence,7,Settings()))

    def test_historical_nano_proof_requires_both_time_brackets_and_good_pixels(self):
        w=NanoWorker('.',mode='guarded',clock=lambda:10.)
        w.identity=7;w.semantic_time=10.01;w.blocked=False
        image=np.zeros((100,200,3),np.uint8)
        a=dict(frame_id=2,visual_time=10.08,box=[90,40,110,60],score=.8,checks=dict(GOOD))
        b=dict(a,frame_id=4,visual_time=10.12)
        frame=Frame(3,10.10,image)
        w.visual_history.append(a);self.assertIsNone(w.evidence_at(frame))
        w.visual_history.append(b);self.assertIsNotNone(w.evidence_at(frame))
        b['checks']['control_suspect']=True;self.assertIsNone(w.evidence_at(frame))
        w.veto('background');self.assertIsNone(w.evidence_at(frame))

    def test_visual_support_repairs_association_with_actual_yolo_not_a_synthetic_box(self):
        tracker=BoTSORTTracker();d=Detection(Box(200,180,260,220),.9,0,'neo')
        for i in range(8):tracker.update([d],10+i*.078,10+i*.078+.12,i,.65,.65)
        identity=tracker.logical_target_id;stamp=10+8*.078
        returning=Detection(Box(255,180,315,220),.9,0,'neo')
        proof=dict(box=[255,180,315,220],frame_id=8,source_time=stamp,identity=identity)
        self.assertTrue(tracker.update([returning],stamp,stamp+.12,8,.65,.65,continuity=proof))
        self.assertEqual(tracker.logical_target_id,identity)
        self.assertEqual(tracker.measurement_time,stamp)
        self.assertEqual(tracker.association_reason,'verified_visual_continuity')
        self.assertEqual(list(vars(tracker.accepted.box).values()),proof['box'])
        old=tracker.measurement_time
        tracker.update([],stamp+.1,stamp+.2,9,.65,.65,continuity=dict(proof,frame_id=9,source_time=stamp+.1))
        self.assertEqual(tracker.measurement_time,old)


class BridgeTransportTests(unittest.TestCase):
    def test_bounded_forward_reaches_mock_wire_and_veto_neutralizes_it(self):
        from follow_neo.manual import ManualControl
        from test_manual import Server, wait_for
        peer=Server();veto=threading.Event();events=[]
        def provider():
            now=time.monotonic();source=now-.28
            d=decision(now,measurement_time=source,measurement_age_ms=280.,
                last_strong_measurement_time=source,accepted=False,state='COAST',
                verified_forward_anchor=dict(time=source,identity=7,forward=.1,allowed=True))
            v=visual(now,semantic_time=source,deadline=source+.4,
                box=[90,40,110,60],checks=dict(GOOD),blocked=veto.is_set(),reason='appearance_unverified')
            return apply_visual(d,v,now)
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(2)))
        control.set_axis_limits(dict(yaw=.5,vertical=.5,roll=.2,forward=.2))
        control.update(set());control.enable();control.start()
        wait_for(lambda:control.enabled);control.start_dance()
        deadline=time.monotonic()+1.5
        while time.monotonic()<deadline and not any(c.startswith('rc ') and float(c.split()[4])>0 for c in peer.commands):
            control.update(set());time.sleep(.01)
        forward=[list(map(float,c.split()[1:])) for c in peer.commands if c.startswith('rc ') and float(c.split()[4])>0]
        self.assertTrue(forward,repr(events))
        self.assertTrue(all(v[1]==v[2]==0 and 0<v[3]<=.2*.35 for v in forward))
        veto.set();control.update(set());count=len(peer.commands)
        wait_for(lambda:'rc 0 0 0 0' in peer.commands[count:])
        self.assertTrue(control.wanted);self.assertEqual(control.mode,'DANCE')


class CandidateSearchTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings();self.r=RecoverySearch()
        self.heading=lambda t:dict(time=t,yaw_deg=0.,source='offline')
        self.r.start_test(10.,self.s,self.heading(10.),'CENTER')
        self.neo=Detection(Box(430,180,490,220),.9,0,'neo')

    def feed(self,stamp,fid,detections=None):
        self.r.observe_candidate([self.neo] if detections is None else detections,stamp,fid,(640,480),self.s)
        return self.r.update(stamp+.12,True,False,'APPROACH',False,self.s,self.heading(stamp+.12),candidate_pending=True)

    def test_two_distinct_raw_frames_steer_bounded_yaw_without_identity_or_deadline_change(self):
        until=self.r.until;source=self.r.source_time
        first=self.feed(10.2,1);duplicate=self.feed(10.2,1)
        self.assertFalse(first['candidate_guidance']);self.assertFalse(duplicate['candidate_guidance'])
        result=self.feed(10.3,2)
        self.assertTrue(result['candidate_guidance']);self.assertGreater(result['scan_yaw'],0)
        self.assertLessEqual(result['scan_yaw'],.20)
        self.assertEqual(self.r.until,until);self.assertEqual(self.r.source_time,source)
        d=decision(10.42,state='DIRECTIONAL_SEARCH',accepted=False,candidate_pending=False,
                   edge_search=result,measurement_time=10.,prediction_time=10.42)
        out=apply_visual(d,visual(10.42,blocked=True,reason='semantic_expired'),10.42)
        cmd,_=dance_command(out,10.42,9.)
        self.assertGreater(cmd[0],0);self.assertEqual(cmd[1:],(0.,0.,0.))

    def test_mini_or_multiple_candidates_or_missing_frames_cancel_guidance(self):
        for ds in ([],[self.neo,Detection(self.neo.box,.8,1,'MINI PRO4')],
                   [self.neo,Detection(Box(60,180,100,220),.9,0,'neo')]):
            self.setUp();self.feed(10.2,1);self.feed(10.3,2)
            self.assertFalse(self.feed(10.4,3,ds)['candidate_guidance'])

    def test_guidance_never_extends_search_budget_and_expires_without_refresh(self):
        self.feed(10.2,1);self.feed(10.3,2);until=self.r.until
        result=self.r.update(10.9,True,False,'APPROACH',False,self.s,self.heading(10.9))
        self.assertFalse(result['candidate_guidance']);self.assertEqual(self.r.until,until)
        for i in range(3,12):self.feed(10.1+i*.1,i)
        self.assertEqual(self.r.until,until)
        result=self.r.update(until+.01,True,False,'APPROACH',False,self.s,self.heading(until+.01))
        self.assertFalse(result['active']);self.assertFalse(result['candidate_guidance'])


if __name__=='__main__':unittest.main()
