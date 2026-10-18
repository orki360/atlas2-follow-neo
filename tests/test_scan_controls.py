"""Offline scan envelopes and bounded reductions on the FIFO control stream."""
import copy
import json
import math
import socket
import time
import unittest
from dataclasses import asdict,replace
from unittest.mock import patch
from follow_neo.types import Settings,settings_from_config
from follow_neo.search_sectors import SectorSweep,SearchObservationLog
from follow_neo.search_motion import SearchYawProfile
from follow_neo.recovery_search import RecoverySearch,scan_command
from follow_neo.control_transport import ControlChannel,MotionReduction,VISUAL_CHANGE,reduced_command
from follow_neo.motion_continuity import compatible_setpoint
import test_motion_continuity as continuity_cases
from test_motion_continuity import heading


class ScanConfigurationTests(unittest.TestCase):
    def test_new_fields_roundtrip_and_old_configs_keep_stop(self):
        s=Settings(scan_observation_degrees=80,scan_yaw_limit=.45,
                   scan_observation_mode='slowdown',scan_observation_yaw=.06)
        self.assertEqual(settings_from_config(dict(schema_version=8,settings=asdict(s))),s)
        self.assertEqual(settings_from_config(dict(schema_version=8,settings={})).scan_observation_mode,'stop')

    def test_invalid_settings_fail_closed(self):
        for key,value in (('scan_observation_degrees',0),('scan_observation_degrees',float('inf')),
                ('scan_yaw_limit',0),('scan_yaw_limit',float('nan')),('scan_yaw_limit',1.1),
                ('scan_observation_yaw',1.01),('scan_observation_yaw',float('inf')),
                ('scan_observation_mode','fly')):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                replace(Settings(),**{key:value}).validate()

    def test_full_circle_and_unrestricted_large_interval_with_sixty_percent(self):
        for angle in (1,360,720,1e6):
            s=Settings(scan_observation_degrees=angle,scan_observation_yaw=.6).validate()
            self.assertEqual(settings_from_config(dict(settings=asdict(s),schema_version=8)),s)
            sweep=SectorSweep(0,0,360,1,step_degrees=angle,observation_mode='slowdown',observation_yaw=.6)
            self.assertEqual(sweep.targets[-1],360)
            if angle>=360:self.assertEqual(sweep.targets,[360])

    def test_custom_steps_preserve_final_partial_sector_and_arc(self):
        self.assertEqual(SectorSweep(0,0,360,1,step_degrees=80).targets,[80,160,240,320,360])
        self.assertEqual(SectorSweep(0,-180,180,-1,step_degrees=80).targets,
                         [-80,-160,-180,-100,-20,60,140,180])
        s=SectorSweep(0,0,360,1,step_degrees=119,observation_mode='slowdown')
        s.index=2
        self.assertEqual(s.target,360)  # the fly-through cannot enlarge the arc

    def test_profile_cap_does_not_rescale_the_calibration_or_boost(self):
        p=SearchYawProfile();p.reset(10,1.)
        for i in range(1,15):
            self.assertLessEqual(p.update(120,0,10+i*.1,max_yaw=.3),.3)
        r=RecoverySearch();s=Settings(search_yaw_degrees=80,scan_yaw_limit=.2)
        r.start_test(10,s,heading(10),'RIGHT',boost=True)
        d=r.update(10.1,True,False,'APPROACH',False,s,heading(10.1))
        self.assertEqual(d['phase'],'BOOST');self.assertEqual(d['direction'],1)

    def test_retarget_and_heading_hold_keep_settings_arc_and_deadline(self):
        s=Settings(search_yaw_degrees=360,scan_observation_degrees=80,
                   scan_yaw_limit=.3,scan_observation_mode='slowdown')
        r=RecoverySearch();r.start_test(10,s,heading(10),'CENTER')
        deadline=r.until;arc=(r.low,r.high)
        r._retarget(1,10.1)
        self.assertEqual(r.sectors.step_degrees,80);self.assertEqual(r.sectors.observation_mode,'slowdown')
        d=r.update(10.2,True,False,'APPROACH',False,s,None)
        self.assertFalse(d['active']);self.assertEqual(d['scan_yaw'],0)
        d=r.update(10.3,True,False,'APPROACH',False,s,heading(10.3,3))
        self.assertEqual(r.until,deadline);self.assertEqual((r.low,r.high),arc)
        self.assertLessEqual(abs(d['scan_yaw']),.3)

    def test_final_scan_gate_honors_independent_cap_and_staleness(self):
        s=Settings(search_yaw_degrees=360,scan_yaw_limit=.2)
        r=RecoverySearch();r.start_test(10,s,heading(10),'RIGHT')
        search=r.update(10.1,True,False,'APPROACH',False,s,heading(10.1,1))
        search['scan_yaw']=.8
        d=dict(edge_search=search,prediction_time=10.1,state='DIRECTIONAL_SEARCH',
               accepted=False,stale=False,spacing_phase='APPROACH')
        self.assertEqual(scan_command(d,10.1,10),.2)
        self.assertEqual(scan_command(d,10.4,10),0)
        for key,value in (('stale',True),('accepted',True),('candidate_pending',True)):
            self.assertEqual(scan_command(dict(d,**{key:value}),10.1,10),0)


class SlowObservationTests(unittest.TestCase):
    def case(self,side=1,**options):
        s=SectorSweep(0,0 if side>0 else -360,360 if side>0 else 0,side,
                      observation_mode='slowdown',settle_seconds=.05,**options)
        p=SearchYawProfile();p.reset(9.9,.08*side)
        return s,p

    def test_slow_pass_keeps_yaw_and_requires_new_results_in_both_directions(self):
        for side in (-1,1):
            s,p=self.case(side)
            self.assertGreater(s.update(side*59,side*6,10,p)*side,0)
            self.assertGreater(s.update(side*59.6,side*6,10.1,p)*side,0)
            self.assertEqual(s.phase,'look')
            s.observe(10.11,1);s.observe(10.21,2)
            yaw=s.update(side*61,side*6,10.3,p)
            self.assertGreater(yaw*side,0);self.assertLessEqual(abs(yaw),.08)
            self.assertEqual(s.index,1);self.assertEqual(len(s.completed),1)
            self.assertEqual(s.completed[0]['target'],side*60)
            self.assertEqual(s.completed[0]['source_frames'],2)

    def test_missing_results_hold_at_window_end_without_advancing(self):
        s,p=self.case();s.update(59,6,10,p);s.update(60,6,10.1,p)
        end=s.target
        for i in range(10):
            self.assertEqual(s.update(end,0,10.2+i*.1,p),0)
        self.assertEqual(s.index,0);self.assertEqual(s.target,end)
        self.assertEqual(s.fallback,'observation_window_end')

    def test_twenty_and_sixty_percent_can_observe_while_turning_above_twelve_deg_s(self):
        for side in (-1,1):
            for cap,rate in ((.2,16.),(.6,45.)):
                s,p=self.case(side,step_degrees=120,observation_yaw=cap)
                s.update(side*119,side*rate,10,p)
                s.update(side*121,side*rate,10.1,p)
                self.assertEqual(s.phase,'look')
                s.observe(10.11,1);s.observe(10.21,2)
                yaw=s.update(side*124,side*rate,10.3,p)
                self.assertGreater(yaw*side,0);self.assertLessEqual(abs(yaw),cap)
                self.assertEqual(s.index,1)
                self.assertEqual(s.completed[-1]['source_frames'],2)

    def test_zero_cap_is_stationary_and_high_cap_cannot_exceed_scan_cap(self):
        s,p=self.case(observation_yaw=0)
        self.assertFalse(s.slow_pass)
        self.assertEqual(s.update(60,0,10,p),0)
        s,p=self.case(observation_yaw=1,yaw_limit=.3)
        self.assertEqual(s.observation_yaw,.3)
        self.assertLessEqual(abs(s.update(60,20,10,p)),.3)

    def test_adaptive_window_never_passes_outer_boundary_or_next_sector(self):
        for step in (1,60,120,360,720):
            for cap in (0,.2,.6,1):
                s,p=self.case(step_degrees=step,observation_yaw=cap)
                self.assertLessEqual(s.target,360)
                if s.slow_pass:self.assertLess(s.target,s.targets[1])

    def test_old_repeated_and_expired_frames_cannot_complete_observation(self):
        s,p=self.case();s.update(59,6,10,p);s.update(60,6,10.1,p)
        s.observe(9.9,1);s.observe(10.11,2);s.observe(10.21,2)
        s.update(61,6,10.3,p);self.assertEqual(s.index,0)
        s.observe(10.22,3);s.update(62,6,10.6,p);self.assertEqual(s.index,0)

    def test_no_completion_before_reaching_configured_angle(self):
        s,p=self.case();s.update(58,6,10,p);s.update(59,6,10.1,p)
        s.observe(10.11,1);s.observe(10.21,2)
        s.update(59.9,6,10.3,p);self.assertEqual(s.index,0)

    def test_blur_rate_or_heading_loss_discards_observation_votes(self):
        for rate in (None,float('nan'),30.,-10.):
            s,p=self.case();s.update(59,6,10,p);s.update(60,6,10.1,p)
            s.observe(10.11,1);s.observe(10.21,2)
            yaw=s.update(61,rate,10.3,p)
            self.assertEqual(s.index,0);self.assertEqual(s.sources,[])
            if rate is None or not math.isfinite(rate) or rate<0:self.assertEqual(yaw,0)

    def test_large_overshoot_cannot_claim_observation_or_reverse(self):
        s,p=self.case()
        self.assertEqual(s.update(80,5,10,p),0)
        self.assertEqual(s.index,0);self.assertEqual(s.fallback,'observation_window_overshoot')

    def test_arc_end_stops_even_in_slowdown_mode(self):
        s,p=self.case();s.index=5
        self.assertEqual(s.update(360,0,10,p),0)
        self.assertEqual(s.update(360,0,10.1,p),0)
        s.observe(10.11,1);s.observe(10.21,2)
        self.assertEqual(s.update(360,0,10.3,p),0)
        self.assertTrue(s.done);self.assertEqual(s.completed[-1]['fallback'],'arc_boundary_stop')

    def test_stop_mode_remains_stationary_and_requires_two_frames(self):
        s=SectorSweep(0,0,360,1,step_degrees=80);p=SearchYawProfile();p.reset(9.9)
        self.assertEqual(s.update(80,0,10,p),0);s.update(80,0,10.1,p)
        s.observe(10.11,1);s.update(80,0,10.3,p);self.assertEqual(s.index,0)
        s.observe(10.31,2);s.update(80,0,10.4,p);self.assertEqual(s.index,1)

    def test_log_reports_transitions_and_durations_without_repeated_ticks(self):
        log=SearchObservationLog();events=[];emit=lambda e,d:events.append((e,d))
        s,p=self.case();s.update(60,6,10,p)
        d=dict(started=9,phase='SCAN',reason='sector_settle',sectors=s.snapshot(),until=30)
        log.update(d,10,emit);log.update(d,10.1,emit)
        s.update(60,6,10.1,p);d.update(reason='sector_look',sectors=s.snapshot())
        log.update(d,10.2,emit);log.update({},10.3,emit)
        self.assertEqual(len(events),3)
        self.assertAlmostEqual(events[1][1]['previous_phase_seconds'],.2)
        json.dumps(events)


class ReductionPermissionTests(unittest.TestCase):
    def test_lower_one_axis_can_preserve_other_authorized_axes(self):
        helper=continuity_cases.PendingPermissionTests();sent,latest=helper.case()
        latest['base_intent']=dict(yaw=.01,vertical=0,roll=0,forward=.1)
        before=copy.deepcopy((sent,latest));status=helper.status(sent,latest)
        reduced=status['reduction_values']
        self.assertGreater(reduced[3],0);self.assertLess(reduced[0],.02)
        self.assertTrue(compatible_setpoint(reduced,(.02,0,0,.0135)))
        self.assertTrue(all(compatible_setpoint(reduced,g) for g in status['allowed_wire_values']))
        self.assertEqual((sent,latest),before)

    def test_reversal_zeros_only_changed_axis_without_creating_opposite_motion(self):
        helper=continuity_cases.PendingPermissionTests();sent,latest=helper.case()
        latest['base_intent']=dict(yaw=-.1,vertical=0,roll=0,forward=.1)
        status=helper.status(sent,latest)
        self.assertEqual(status['reduction_values'][0],0)
        self.assertGreater(status['reduction_values'][3],0)

    def test_identity_background_proximity_and_semantic_vetoes_never_offer_reduction(self):
        changes=[('nano',dict(blocked=True,reason='appearance_or_background_drift')),
                 ('nano',dict(blocked=True,reason='competing_mini_detection')),
                 ('nano',dict(identity=77)),('nano',dict(epoch=77)),('nano',dict(generation=2)),
                 ('nano',dict(mode='shadow')),('decision',dict(stale=True)),
                 ('decision',dict(spacing_close_guard_armed=True)),('decision',dict(bbox_clipped=True)),
                 ('decision',dict(spacing_phase='STOPPED'))]
        for where,change in changes:
            helper=continuity_cases.PendingPermissionTests();sent,latest=helper.case()
            latest['base_intent']=dict(yaw=.01,vertical=0,roll=0,forward=.1)
            (latest['nano'] if where=='nano' else latest).update(change)
            self.assertNotIn('reduction_values',helper.status(sent,latest),(where,change))
        sent,latest=helper.case();sent['nano']['semantic_time']=9.
        self.assertNotIn('reduction_values',helper.status(sent,latest))

    def test_transport_rejects_increases_reversals_nonfinite_and_new_axes(self):
        for v in ((.3,0,0,.1),(-.1,0,0,.1),(.1,.1,0,.1),(.1,0,0,float('nan')),
                  (0,0,0,0),(.2,0,0,.1)):
            self.assertIsNone(reduced_command('rc .2 0 0 .1',MotionReduction(v)),v)


class ReductionTransportTests(unittest.TestCase):
    def exchange(self,*,veto=False,no_reply=False,second_reduction=False,reject=False):
        now=[10.];current=[(.2,0.,0.,.1)];events=[];changes=[]
        class Peer:
            def __init__(self):self.wire=[];self.replied=0
            def setsockopt(self,*args):pass
            def settimeout(self,value):self.timeout=value
            def sendall(self,data):self.wire.append(data)
            def recv(self,n):
                now[0]+=self.timeout
                if not no_reply and now[0]>=10.12 and self.replied<len(self.wire):
                    count=len(self.wire)-self.replied;self.replied=len(self.wire)
                    return (b'denied\r\n' if reject else b'success\r\n')*count
                raise socket.timeout()
        def guard(keys):
            if keys is None:return None
            if veto and now[0]>=10.08:return VISUAL_CHANGE
            if now[0]>=10.04 and current[0][0]>.1:return MotionReduction((.1,0,0,.1))
            if second_reduction and now[0]>=10.08:return MotionReduction((.05,0,0,.1))
        def applied(values):current[0]=values;changes.append(values)
        peer=Peer()
        with patch('follow_neo.control_transport.time.monotonic',side_effect=lambda:now[0]):
            channel=ControlChannel(peer,guard,lambda e,d:events.append((e,d)))
            ok=channel.command('rc .2 0 0 .1',motion_keys=frozenset(),on_reduction=applied)
        return channel,peer,ok,events,changes

    def test_one_reduction_is_fifo_acked_without_full_neutral(self):
        c,p,ok,events,changes=self.exchange()
        self.assertTrue(ok);self.assertFalse(c.pending);self.assertEqual(len(p.wire),2)
        self.assertEqual(p.wire[-1],b'rc 0.1000 0.0000 0.0000 0.1000\r\n')
        self.assertEqual(changes,[(.1,0,0,.1)])
        acks=[d for e,d in events if e=='control_command_ack']
        self.assertEqual([d['sequence'] for d in acks],[1,2])
        self.assertEqual(acks[-1]['command'],p.wire[-1].decode().strip())

    def test_background_after_reduction_still_sends_separate_neutral(self):
        c,p,ok,events,changes=self.exchange(veto=True)
        self.assertFalse(ok);self.assertTrue(c.neutral_acknowledged)
        self.assertEqual(p.wire[-1],b'rc 0 0 0 0\r\n');self.assertFalse(c.pending)

    def test_second_reduction_is_not_an_unbounded_command_queue(self):
        c,p,ok,events,changes=self.exchange(second_reduction=True)
        self.assertFalse(ok);self.assertEqual(len(changes),1)
        self.assertEqual(p.wire[-1],b'rc 0 0 0 0\r\n')

    def test_missing_ack_uses_original_deadline_and_neutral(self):
        c,p,ok,events,changes=self.exchange(no_reply=True)
        self.assertFalse(ok);self.assertTrue(c.paused);self.assertIn('0.25s',c.pause_reason)
        stop=next(d for e,d in events if e=='control_paused')
        reduction=next(d for e,d in events if e=='control_motion_reduced')
        self.assertEqual(reduction['deadline'],10.25)
        self.assertFalse(reduction['ack_deadline_extended'])
        self.assertEqual(p.wire[-1],b'rc 0 0 0 0\r\n')

    def test_rejected_reply_cannot_be_counted_as_success(self):
        c,p,ok,events,changes=self.exchange(reject=True)
        self.assertFalse(ok);self.assertTrue(c.paused);self.assertTrue(c.rejected)

    def test_before_send_reduction_logs_only_actual_wire_command(self):
        wire=[];events=[];applied=[]
        class Peer:
            def setsockopt(self,*a):pass
            def settimeout(self,t):pass
            def sendall(self,data):wire.append(data)
            def recv(self,n):return b'success\r\n'
        def guard(keys):
            return MotionReduction((.1,0,0,.1)) if keys is not None and not applied else None
        c=ControlChannel(Peer(),guard,lambda e,d:events.append((e,d)))
        self.assertTrue(c.command('rc .2 0 0 .1',motion_keys=frozenset(),on_reduction=applied.append))
        self.assertEqual(wire,[b'rc 0.1000 0.0000 0.0000 0.1000\r\n'])
        self.assertTrue(next(d for e,d in events if e=='control_motion_reduced')['before_send'])


class ManualReductionIntegrationTests(unittest.TestCase):
    def test_reduction_reaches_wire_and_logs_without_restarting_forward(self):
        from follow_neo.manual import ManualControl
        from follow_neo.nano_guard import apply_visual
        from test_nano_bridge import decision,visual
        from test_visual_continuity import GOOD
        from test_manual import Server,wait_for
        events=[];first_motion=[None];origin=time.monotonic()
        def delay(cmd):
            if cmd.startswith('rc ') and any(float(v) for v in cmd.split()[1:]):
                if first_motion[0] is None:
                    first_motion[0]=time.monotonic();time.sleep(.11)
        peer=Server(before_reply=delay)
        def provider():
            now=time.monotonic();frame=math.floor((now-origin)/.1);source=origin+frame*.1-.12
            yaw=.001 if first_motion[0] is not None else .3
            d=decision(now,measurement_time=source,last_strong_measurement_time=source,
                measurement_age_ms=(now-source)*1000,measurement_id=frame,
                motion_settings=dict(nano_visual_max_age_seconds=.07),
                intent=dict(yaw=yaw,vertical=0.,roll=0.,forward=.1))
            v=visual(now,semantic_time=source,visual_time=now-.04,deadline=now+.06,checks=dict(GOOD))
            return apply_visual(d,v,now)
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.5,vertical=.1,roll=.1,forward=.09))
        control.update(set());control.enable();control.start();wait_for(lambda:control.enabled)
        control.start_dance();end=time.monotonic()+.7
        while time.monotonic()<end:control.update(set());time.sleep(.01)
        reductions=[d for e,d in events if e=='control_motion_reduced']
        self.assertEqual(len(reductions),1)
        sent=[float(x) for x in reductions[0]['command'].split()[1:]]
        self.assertGreater(sent[3],0)
        commands=[d for e,d in events if e=='flight_command' and d.get('motion_reduced')]
        self.assertEqual(len(commands),1);self.assertEqual(commands[0]['values'],sent)
        self.assertFalse(any(e=='control_motion_neutralizing' for e,d in events))


if __name__=='__main__':unittest.main()
