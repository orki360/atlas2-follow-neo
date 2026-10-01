import unittest
from unittest.mock import patch
import numpy as np
from follow_neo.approach import LateralEvidence,forward_diagnostics,forward_status_text
from follow_neo.centering import DampedTrackingController
from follow_neo.types import Settings,Kinematics,Box,Detection
from follow_neo.controller import FollowController
from follow_neo.dance import DanceSmoother
from follow_neo.vendor.botsort.kalman_filter import KalmanFilter,positive_covariance


def observation(i,x,warp=0,valid=True):
    return dict(time=10+i*.04,frame_id=i,point=(x,240),width=640,
                warp=[[1,0,warp],[0,1,0]],camera_valid=valid)


class ApproachTests(unittest.TestCase):
    def test_final_smoother_does_not_carry_roll_when_evidence_is_removed(self):
        s=DanceSmoother()
        for i in range(10):s.update((0,0,.5,1),10+i*.05,damped_lateral=True)
        self.assertGreater(s.values[2],0)
        self.assertEqual(s.update((0,0,0,1),10.46,damped_lateral=True)[2],0)

    def test_camera_pan_alone_never_commands_roll(self):
        e=LateralEvidence()
        for i in range(20):
            o=observation(i,320+i*8,8)
            self.assertEqual(e.update(o,o['time'],True,''),0)

    def test_consistent_relative_motion_requires_new_measurements(self):
        e=LateralEvidence();o=observation(0,320)
        for j in range(4):self.assertEqual(e.update(o,10+j*.02,True,''),0)
        for i in range(1,4):
            o=observation(i,320+i*4);self.assertEqual(e.update(o,o['time'],True,''),0)
        o=observation(4,336);self.assertGreater(e.update(o,o['time'],True,''),0)
        for i in range(5,12):
            o=observation(i,320+i*4);self.assertLessEqual(e.update(o,o['time'],True,''),.1)
        self.assertEqual(e.update(o,10.5,False,'align_yaw_first'),0)

    def test_jitter_gaps_and_camera_failure_do_not_accumulate_evidence(self):
        e=LateralEvidence()
        for i in range(12):
            o=observation(i,320+(-1)**i*4);self.assertEqual(e.update(o,o['time'],True,''),0)
        for i in range(12,20):
            o=observation(i,320+i*4,valid=False);self.assertEqual(e.update(o,o['time'],True,''),0)
        for i in range(20,30,2):
            o=observation(i,320+i*4);self.assertEqual(e.update(o,o['time'],True,''),0)

    def test_centered_static_target_keeps_forward_without_roll_despite_predicted_vx(self):
        c=DampedTrackingController();s=Settings()
        for i in range(15):
            t=10+i*.04;k=Kinematics(True,320,240,40,30,300,0)
            out,_,_=c.compute(k,640,480,0,False,t,s,dict(time=t,yaw_deg=0,source='test'),t,0,observation(i,320))
            self.assertEqual(out.roll,0);self.assertEqual(out.forward,1)
        self.assertEqual(c.forward_factor,1)

    def test_misalignment_or_missing_heading_blocks_measured_lateral_motion(self):
        for ex,rate_available in ((.4,True),(.1,False)):
            c=DampedTrackingController()
            for i in range(12):
                t=10+i*.04;k=Kinematics(True,320+320*ex,240,40,30,300,0)
                h=dict(time=t,yaw_deg=0,source='test') if rate_available else None
                out,_,_=c.compute(k,640,480,0,False,t,Settings(),h,t,ex,observation(i,320+4*i))
                self.assertEqual(out.roll,0)

    def test_status_distinguishes_caps_alignment_distance_and_loss(self):
        args=['TRACK','APPROACH','approach',1,1,True,False,1]
        d=forward_diagnostics(*args)
        self.assertIn('3.0%',forward_status_text({'forward_control':d},.03))
        self.assertEqual(d['reason'],'Using configured forward speed')
        args[4]=.2;self.assertEqual(forward_diagnostics(*args)['reason'],'Aligning with target')
        args[3]=.2;self.assertEqual(forward_diagnostics(*args)['reason'],'Slowing for distance')
        args[0]='REACQUIRE';self.assertEqual(forward_diagnostics(*args)['reason'],'Waiting for verified target')


class NumericalTests(unittest.TestCase):
    def test_joseph_update_matches_well_conditioned_reference(self):
        k=KalmanFilter();mean,cov=k.initiate(np.array([100.,150.,60.,40.]))
        mean,cov=k.predict(mean,cov);z=np.array([105.,152.,58.,41.]);m,s=k.project(mean,cov)
        gain=np.linalg.solve(s,(cov@k._update_mat.T).T).T
        actual,p=k.update(mean,cov,z)
        np.testing.assert_allclose(actual,mean+gain@(z-m))
        np.testing.assert_allclose(p,cov-gain@s@gain.T,atol=1e-10)

    def test_large_prediction_covariance_correction_stays_positive(self):
        k=KalmanFilter();mean=np.array([100.,150.,60.,40.,0,0,0,0])
        cov=np.eye(8)*1e17
        for i in range(100):
            mean,cov=k.update(mean,cov,np.array([100.+i*.1,150.,60.,40.]))
            np.linalg.cholesky(cov)
            mean,cov=k.predict(mean,cov)
        self.assertTrue(np.isfinite(cov).all())

    def test_roundoff_repair_does_not_hide_real_corruption(self):
        p=np.eye(8);p[0,0]=-1e-13;np.linalg.cholesky(positive_covariance(p))
        for bad in (-1.,float('nan')):
            p=np.eye(8);p[0,0]=bad
            with self.assertRaises(np.linalg.LinAlgError):positive_covariance(p)

    def test_tracker_failure_zeroes_control_then_reacquires_fresh_target(self):
        c=FollowController(tracker_backend='botsort');s=Settings(confidence=.8,new_track_confidence=.8)
        det=[Detection(Box(290,220,350,260),.95)]
        for i in range(8):
            t=10+i*.04;c.observe(det,t,t,i,(640,480),s);d=c.tick(t,640,480,t,s)
        with patch.object(c.tracker.engine,'update',side_effect=np.linalg.LinAlgError('2-th leading minor')):
            self.assertFalse(c.observe(det,10.4,10.4,10,(640,480),s))
        d=c.tick(10.4,640,480,10.4,s)
        self.assertEqual(list(d['intent'].values()),[0,0,0,0]);self.assertFalse(d['confirmed'])
        self.assertEqual(d['tracking']['numerical_recoveries'],1)
        self.assertIsNotNone(d['tracking']['numerical_event'])
        self.assertFalse(c.observe(det,10.4,10.41,10,(640,480),s))
        for i in range(1,8):
            t=10.4+i*.04;c.observe(det,t,t,10+i,(640,480),s);d=c.tick(t,640,480,t,s)
            if i==1:self.assertEqual(d['intent']['forward'],0)
        self.assertEqual(d['state'],'TRACK');self.assertTrue(d['confirmed'])
        self.assertGreater(d['intent']['forward'],0);self.assertEqual(d['tracking']['numerical_recoveries'],1)

    def test_other_errors_are_not_silently_treated_as_covariance_recovery(self):
        c=FollowController(tracker_backend='botsort');s=Settings()
        c.observe([],10,10,1,(640,480),s)
        with patch.object(c.tracker.engine,'update',side_effect=ValueError('programming error')):
            with self.assertRaises(ValueError):c.observe([],10.1,10.1,2,(640,480),s)


if __name__=='__main__':unittest.main()
