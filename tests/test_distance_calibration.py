"""Measured-only range fitting and the real Tk collect/save/check workflow."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.distance_calibration import (SampleCapture,fit_profile,load_profile,load_calibrations,fit_global_profile,load_global_profile,measured_mini,estimate_distance,TARGET_VIEWS)
from follow_neo.video import Frame,LatestValue
from follow_neo.roi import RoiVerifier,RoiSettings
from follow_neo.detector import DetectorSettings

CONTEXT=dict(resolution=[1920,1080],model='test-model',source='test-camera')

def point(d,w,n=100):
    return dict(distance_m=d,width_px=w,samples=n,span_seconds=9.,relative_spread=.02)

def profile():return fit_profile([point(2,300),point(4,150),point(6,100)],CONTEXT,'Camera / 1x / front')

def result(fid=1,stamp=10,width=150,**changes):
    box=[960-width/2,510,960+width/2,570]
    det=dict(model_class_name='MINI PRO4',confidence=.9,tracking_eligible=False,box=box)
    det.update(changes)
    return dict(frame=Frame(fid,stamp,np.zeros((1080,1920,3),np.uint8)),detections=[det],generation=0)


def verified_result(verifier,fid,stamp,width=150,direct=False,local_name='MINI PRO4'):
    raw=result(fid,stamp,width,model_class_name='MINI PRO4' if direct else 'NEO',tracking_eligible=not direct)
    detector=SimpleNamespace(class_names={0:'NEO',1:'MINI PRO4'},settings=DetectorSettings())
    # ROI returns a deliberately narrower box. Calibration must use the full
    # frame's width, not this crop-local box or its translated equivalent.
    detector.detect=lambda image:dict(detections=[dict(box=[480-width*.4,246,480+width*.4,294],
        confidence=.9,model_class_name=local_name,class_id=1 if local_name=='MINI PRO4' else 0)])
    return verifier.process(raw['frame'],raw,detector,RoiSettings())


class DistanceCalibrationTests(unittest.TestCase):
    def test_fit_and_equal_station_weight_not_capture_length(self):
        p=profile();self.assertEqual(p['k_m_px'],600)
        self.assertEqual(p['fit_max_error'],0)
        p=fit_profile([point(2,300,20),point(4,150,20),point(6,101,500)],CONTEXT,'test')
        self.assertEqual(p['k_m_px'],600)
    def test_invalid_or_inconsistent_calibration_cannot_be_saved(self):
        for points in ([point(2,300)], [point(2,300)]*3,
                       [point(2,300),point(2.1,285),point(2.2,273)],
                       [point(2,300),point(4,150),point(6,300)],
                       [point(0,300),point(4,150),point(6,100)],
                       [point(2,float('nan')),point(4,150),point(6,100)]):
            with self.subTest(points=points),self.assertRaises(ValueError):fit_profile(points,CONTEXT,'test')
    def test_reload_recomputes_coefficients_and_check_error(self):
        p=profile();p['k_m_px']=9000;p['validations']=[dict(distance_m=3,width_px=200,error_m=99)]
        loaded=load_profile(json.loads(json.dumps(p)))
        self.assertEqual(loaded['k_m_px'],600)
        self.assertEqual(loaded['validations'][0]['relative_error'],0)
        with self.assertRaises(ValueError):load_profile({'schema_version':90})
    def test_only_fresh_unique_mini_full_source_boxes_are_used(self):
        r=result();r['full_detections']=[dict(box=[0,0,1,1])]
        m,_=measured_mini(r,10.1);self.assertEqual(m['width_px'],150)
        self.assertIsNone(measured_mini(r,10.3)[0])
        for changes in (dict(model_class_name='NEO'),dict(confidence=.49),dict(confidence=float('nan')),
                        dict(confidence=float('inf')),dict(box=[0,500,100,560]),dict(box=[100,510,200,570]),
                        dict(box=[955,538,965,542]),dict(box=[float('nan'),0,30,50])):
            with self.subTest(changes=changes):self.assertIsNone(measured_mini(result(**changes),10.1)[0])
        r=result();r['detections']*=2;self.assertIsNone(measured_mini(r,10.1)[0])
        r=result();r['detections']=[];r['track_box']=[700,500,1000,560]
        self.assertIsNone(measured_mini(r,10.1)[0])
    def test_neo_alongside_mini_is_ignored_and_tracking_eligibility_is_unchanged(self):
        r=result()
        neo=result(width=300,model_class_name='NEO',tracking_eligible=True)['detections'][0]
        r['detections'].append(neo)
        before=deepcopy(r['detections'])
        measurement,_=measured_mini(r,10.1)
        self.assertEqual(measurement['width_px'],150)
        self.assertEqual(r['detections'],before)
        self.assertFalse(r['detections'][0]['tracking_eligible'])
        self.assertTrue(r['detections'][1]['tracking_eligible'])
    def test_roi_reclassified_mini_is_measurable_without_tracking_authority(self):
        r=result(roi_status='rejected_mini',roi_id=7)
        measurement,_=measured_mini(r,10.1)
        self.assertEqual(measurement['width_px'],150)
        self.assertFalse(r['detections'][0]['tracking_eligible'])
    def test_old_neo_and_wrong_target_profiles_cannot_supply_mini_distance(self):
        for changes in (dict(schema_version=1,neo_view='front'),dict(target_class='NEO'),dict(target_class=None)):
            old={**profile(),**changes}
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):load_profile(old)
                self.assertIsNone(estimate_distance(old,measured_mini(result(),10.1)[0],CONTEXT)[0])
    def test_capture_uses_new_frames_and_rejects_position_jump(self):
        c=SampleCapture(4,10,CONTEXT)
        for i in range(100):
            m,_=measured_mini(result(i+1,10+i*.09,roi_id=1),10+i*.09)
            self.assertTrue(c.add(m));self.assertFalse(c.add(m))
        p=c.finish();self.assertEqual(p['samples'],100);self.assertEqual(p['width_px'],150)
        m,_=measured_mini(result(101,19.1,box=[1100,510,1250,570]),19.1)
        with self.assertRaises(ValueError):c.add(m)
    def test_real_roi_correction_uses_full_frame_width_and_remains_display_only(self):
        r=verified_result(RoiVerifier(),1,10.)
        self.assertEqual(r['full_detections'][0]['model_class_name'],'NEO')
        self.assertEqual(r['detections'][0]['roi_status'],'rejected_mini')
        self.assertFalse(r['detections'][0]['tracking_eligible'])
        local=r['rois'][0]['detections'][0]['box']
        self.assertEqual(local[2]-local[0],120)
        m,reason=measured_mini(r,10.1)
        self.assertEqual(m['width_px'],150)
        self.assertEqual(m['classification_source'],'roi')
        self.assertIn('ROI',reason)
        self.assertEqual(estimate_distance(profile(),m,CONTEXT)[0],4)
    def test_roi_neo_and_unresolved_classifications_never_supply_mini_samples(self):
        for name in ('NEO','unknown'):
            with self.subTest(name=name):
                self.assertIsNone(measured_mini(verified_result(RoiVerifier(),1,10.,local_name=name),10.1)[0])
    def test_capture_rejects_long_detection_gap(self):
        capture=SampleCapture(4,10,CONTEXT)
        capture.add(measured_mini(result(1,10.),10.)[0])
        with self.assertRaises(ValueError):capture.add(measured_mini(result(2,11.),11.)[0])
    def test_capture_rejects_short_or_moving_samples(self):
        for unstable in (False,True):
            c=SampleCapture(4,10,CONTEXT)
            for i in range(25):
                stamp=10+i*(.3 if unstable else .01)
                m,_=measured_mini(result(i+1,stamp,width=150+i*4 if unstable else 150),stamp);c.add(m)
            with self.assertRaises(ValueError):c.finish()
    def test_context_changes_and_extrapolation_are_explicit(self):
        m,_=measured_mini(result(width=150),10.)
        self.assertEqual(estimate_distance(profile(),m,CONTEXT)[0],4)
        self.assertIsNone(estimate_distance(profile(),m,{**CONTEXT,'model':'other'})[0])
        self.assertIsNone(estimate_distance(profile(),m,{**CONTEXT,'resolution':[960,540]})[0])
        self.assertIsNone(estimate_distance(None,m,CONTEXT)[0])
        m['width_px']=50;d,quality=estimate_distance(profile(),m,CONTEXT)
        self.assertEqual(d,12);self.assertIn('Outside',quality)
    def test_single_side_migration_preserves_points_and_validation(self):
        old=fit_profile(profile()['points'],CONTEXT,'test',target_view='right')
        old['validations']=[dict(distance_m=3,width_px=200)]
        loaded=load_calibrations(old)
        self.assertEqual(loaded['selected_view'],'right')
        self.assertEqual(set(loaded['profiles']),{'right'})
        self.assertEqual(loaded['profiles']['right']['points'],old['points'])
        self.assertEqual(loaded['profiles']['right']['validations'][0]['relative_error'],0)
        self.assertEqual(load_calibrations(json.loads(json.dumps(loaded))),loaded)
    def test_bundle_rejects_mismatched_or_unknown_sides_and_old_neo(self):
        p=fit_profile(profile()['points'],CONTEXT,'test',target_view='left')
        good=dict(schema_version=3,target_class='MINI PRO4',selected_view='left',profiles={'left':p})
        for bad in ({**good,'profiles':{'right':p}},{**good,'profiles':{'diagonal':p}},
                    {**good,'selected_view':'unknown'},{**good,'target_class':'NEO'},
                    {**p,'schema_version':1},profile()):
            with self.subTest(bad=bad),self.assertRaises(ValueError):load_calibrations(bad)



class GlobalCalibrationTests(unittest.TestCase):
    def mixed_points(self):
        return [point(d,k/d) for d in (2,4,6) for k in (360,400,600,660)]

    def test_global_fit_pools_angles_and_exposes_observed_spread(self):
        p=fit_global_profile(self.mixed_points(),CONTEXT)
        self.assertEqual(p['k_m_px'],500)
        self.assertEqual((p['k_min_m_px'],p['k_max_m_px']),(360,660))
        self.assertEqual(p['mode'],'global')
        self.assertNotIn('target_view',p);self.assertNotIn('profiles',p)
        self.assertGreater(p['fit_max_error'],.25)
        measurement=measured_mini(result(width=150),10.)[0]
        distance,quality=estimate_distance(p,measurement,CONTEXT)
        self.assertAlmostEqual(distance,500/150)
        self.assertIn('2.40-4.40 m',quality)
        self.assertIn('Within calibrated range',quality)

    def test_distances_have_equal_weight_despite_repeated_captures(self):
        points=[point(2,500)]*40+[point(4,150),point(6,100)]
        p=fit_global_profile(points,CONTEXT)
        self.assertEqual(p['k_m_px'],600)
        self.assertEqual(p['k_max_m_px'],1000)

    def test_multiple_angles_at_one_distance_do_not_replace_distance_coverage(self):
        for points in ([point(3,w) for w in (120,140,180,210)],
                       [point(3,150),point(3.1,160),point(3.2,170)]):
            with self.assertRaises(ValueError):fit_global_profile(points,CONTEXT)

    def test_global_reload_recomputes_coefficients_envelope_and_validation(self):
        p=fit_global_profile(self.mixed_points(),CONTEXT)
        p.update(k_m_px=99999,k_min_m_px=1,k_max_m_px=2)
        p['validations']=[dict(distance_m=3,width_px=200,estimated_m=0)]
        loaded=load_global_profile(json.loads(json.dumps(p)))
        self.assertEqual((loaded['k_m_px'],loaded['k_min_m_px'],loaded['k_max_m_px']),(500,360,660))
        self.assertEqual(loaded['validations'][0]['estimated_m'],2.5)
        self.assertEqual(loaded['validations'][0]['error_m'],-.5)

    def test_saved_side_profiles_merge_without_losing_points(self):
        profiles={}
        for view,k in zip(TARGET_VIEWS,(360,400,600,660)):
            p=fit_profile([point(d,k/d) for d in (2,4,6)],CONTEXT,'test',target_view=view)
            p['validations']=[dict(distance_m=3,width_px=k/3)]
            profiles[view]=p
        bundle=dict(schema_version=3,target_class='MINI PRO4',selected_view='left',profiles=profiles)
        loaded=load_global_profile(bundle)
        self.assertEqual(len(loaded['points']),12)
        self.assertEqual(loaded['k_m_px'],500)
        self.assertEqual(len(loaded['validations']),4)
        self.assertEqual(loaded['points'],[point for p in profiles.values() for point in p['points']])
        self.assertEqual(load_global_profile(json.loads(json.dumps(loaded))),loaded)

    def test_migration_rejects_different_sources_and_old_neo_profiles(self):
        front=fit_profile(profile()['points'],CONTEXT,'test',target_view='front')
        left=fit_profile(profile()['points'],{**CONTEXT,'source':'another-phone'},'test',target_view='left')
        bundle=dict(schema_version=3,target_class='MINI PRO4',selected_view='front',profiles=dict(front=front,left=left))
        with self.assertRaisesRegex(ValueError,'different camera'):load_global_profile(bundle)
        for old in ({**front,'target_class':'NEO'},{**front,'schema_version':1}):
            with self.assertRaises(ValueError):load_global_profile(old)

    def test_single_side_migrates_into_global_points(self):
        old=fit_profile(profile()['points'],CONTEXT,'test',target_view='rear')
        old['validations']=[dict(distance_m=3,width_px=200)]
        loaded=load_global_profile(old)
        self.assertEqual(loaded['points'],old['points'])
        self.assertEqual(loaded['mode'],'global')
        self.assertEqual(loaded['validations'][0]['relative_error'],0)


class DistancePanelTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from tkinter import ttk
        from follow_neo.distance_view import DistancePanel
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        try:self.root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        self.root.withdraw();self.addCleanup(self.root.destroy)
        self.path=Path(self.tmp.name)/'config.json'
        self.session=SimpleNamespace(generation=0,result=LatestValue(),model_manifest={'sha256':'test-model'},receiver=SimpleNamespace(host='test-camera'),log=Mock())
        self.owner=SimpleNamespace(root=self.root,tabs=ttk.Notebook(self.root),distance_text=tk.StringVar(),session=self.session,stopping=False)
        self.owner.save_config=lambda:self.path.write_text(json.dumps({'distance_calibration':self.owner.distance_calibration}),encoding='utf-8')
        self.panel=DistancePanel(self.owner,{})
        self.session.result.set(result(stamp=10.));self.panel.update(10.)
        self.fid=1

    def collect(self,distance,width,start,kind='calibration'):
        self.panel.distance.set(str(distance));self.session.result.set(result(self.fid,start,width));self.fid+=1
        with patch('follow_neo.distance_view.time.monotonic',return_value=start):self.panel.start(kind)
        self.assertIsNotNone(self.panel.capture)
        for i in range(1,101):
            stamp=start+i*.1
            self.session.result.set(result(self.fid,stamp,width));self.fid+=1
            self.panel.update(stamp)
        self.assertIsNone(self.panel.capture)

    def test_collect_save_validate_reload_and_stale_display(self):
        self.collect(2,300,10);self.collect(4,150,21);self.collect(6,100,32)
        self.assertEqual(len(self.panel.points),3)
        self.assertIsNone(self.owner.distance_calibration)
        self.panel.save();self.assertEqual(self.owner.distance_calibration['k_m_px'],600)
        self.collect(3,200,43,'validation')
        p=load_global_profile(json.loads(self.path.read_text())['distance_calibration'])
        self.assertEqual(p['validations'][-1]['estimated_m'],3)
        self.assertIn('~3.00 m',self.owner.distance_text.get())
        self.panel.update(54.);self.assertIn('stale',self.owner.distance_text.get())
        from follow_neo.distance_view import DistancePanel
        second=DistancePanel(self.owner,json.loads(self.path.read_text()))
        self.assertEqual(len(second.points),3)
        self.session.result.set(result(self.fid,55.,200));second.update(55.)
        self.assertIn('~3.00 m',self.owner.distance_text.get())

    def test_collect_save_with_alternating_direct_and_real_roi_corrected_mini(self):
        verifier=RoiVerifier()
        for distance,width,start in ((2,300,10.),(4,150,21.),(6,100,32.)):
            self.panel.distance.set(str(distance))
            self.session.result.set(verified_result(verifier,self.fid,start,width));self.fid+=1
            with patch('follow_neo.distance_view.time.monotonic',return_value=start):self.panel.start('calibration')
            self.assertIsNotNone(self.panel.capture)
            for i in range(1,101):
                stamp=start+i*.1
                self.session.result.set(verified_result(verifier,self.fid,stamp,width,direct=i%2==1));self.fid+=1
                self.panel.update(stamp)
                if i<100:self.assertIsNotNone(self.panel.capture,self.panel.status.get())
            self.assertIsNone(self.panel.capture)
            self.assertIn('ROI',self.panel.measurement_status.get())
        self.assertEqual([p['width_px'] for p in self.panel.points],[300,150,100])
        self.panel.save()
        saved=load_global_profile(json.loads(self.path.read_text())['distance_calibration'])
        self.assertEqual(saved['k_m_px'],600)
        self.assertEqual(saved['target_class'],'MINI PRO4')

    def test_collect_all_angles_into_one_table_save_once_and_reload(self):
        self.assertFalse(hasattr(self.panel,'view_selector'))
        self.assertFalse(hasattr(self.panel,'target_view'))
        stamp=10.
        for distance in (2,4,6):
            for k in (360,400,600,660):
                self.collect(distance,k/distance,stamp);stamp+=11
        self.assertEqual(len(self.panel.points),12)
        self.assertEqual(len(self.panel.table.get_children()),12)
        self.panel.save()
        saved=json.loads(self.path.read_text())['distance_calibration']
        self.assertEqual(saved['schema_version'],4)
        self.assertEqual(saved['mode'],'global')
        self.assertEqual(saved['k_m_px'],500)
        self.assertNotIn('profiles',saved)
        from follow_neo.distance_view import DistancePanel
        self.panel=DistancePanel(self.owner,{'distance_calibration':saved})
        self.assertEqual(len(self.panel.points),12)
        self.collect(3,200,stamp,'validation')
        self.assertIn('~2.50 m',self.owner.distance_text.get())
        self.assertIn('1.80-3.30 m',self.owner.distance_text.get())
        self.assertEqual(len(self.panel.profile['points']),12)
        self.assertEqual(self.panel.profile['validations'][0]['error_m'],-.5)

    def test_old_side_calibration_becomes_global_and_accepts_other_angles(self):
        from follow_neo.distance_view import DistancePanel
        old=fit_profile(profile()['points'],CONTEXT,'test',target_view='rear')
        old['validations']=[dict(distance_m=3,width_px=200)]
        self.panel=DistancePanel(self.owner,{'distance_calibration':old})
        self.assertEqual(self.panel.points,old['points'])
        self.assertEqual(self.panel.profile['mode'],'global')
        self.assertEqual(len(self.panel.profile['validations']),1)
        self.collect(4,90,10.)
        self.assertEqual(len(self.panel.points),4)
        self.panel.save()
        saved=load_global_profile(json.loads(self.path.read_text())['distance_calibration'])
        self.assertEqual(len(saved['points']),4)
        self.assertNotIn('target_view',saved)

    def test_save_failure_keeps_previous_profile_and_camera_change_cancels(self):
        self.panel.points=profile()['points'];self.panel.context=CONTEXT
        self.panel.save();previous=self.owner.distance_calibration
        self.owner.save_config=Mock(side_effect=OSError('disk unavailable'))
        self.panel.points=[point(2,310),point(4,155),point(6,103)]
        with patch('follow_neo.distance_view.messagebox.showerror') as error:
            self.panel.save();error.assert_called_once()
        self.assertIs(self.owner.distance_calibration,previous)
        self.assertIs(self.panel.profile,previous)
        self.panel.distance.set('3');self.session.result.set(result(self.fid,10.))
        with patch('follow_neo.distance_view.time.monotonic',return_value=10.):self.panel.start('validation')
        self.assertIsNotNone(self.panel.capture)
        self.session.generation=1;self.panel.update(10.1)
        self.assertIsNone(self.panel.capture)

    def test_draft_editing_keeps_saved_global_profile_until_save(self):
        self.panel.points=profile()['points'];self.panel.context=CONTEXT;self.panel.save()
        saved=deepcopy(self.owner.distance_calibration)
        self.panel.table.selection_set('0');self.panel.remove()
        self.assertEqual(len(self.panel.points),2)
        self.panel.clear();self.assertEqual(self.panel.points,[])
        self.assertEqual(self.owner.distance_calibration,saved)
        self.panel.update(10.)
        self.assertIn('~4.00 m',self.owner.distance_text.get())

    def test_invalid_legacy_data_is_not_erased_by_unrelated_settings_save(self):
        from follow_neo.distance_view import DistancePanel
        old=dict(schema_version=1,target_class='NEO',points=[{'kept':'original'}])
        panel=DistancePanel(self.owner,{'distance_calibration':old})
        self.assertIsNone(panel.profile)
        self.assertIs(self.owner.distance_calibration,old)
        self.assertIn('could not be loaded',panel.status.get())

    def test_capture_cancel_and_invalid_distance_do_not_add_points(self):
        self.panel.distance.set('nan')
        with patch('follow_neo.distance_view.time.monotonic',return_value=10.),patch('follow_neo.distance_view.messagebox.showerror') as error:
            self.panel.start('calibration');error.assert_called_once()
        self.assertIsNone(self.panel.capture)
        self.panel.distance.set('3')
        with patch('follow_neo.distance_view.time.monotonic',return_value=10.):self.panel.start('calibration')
        self.panel.cancel();self.assertEqual(self.panel.points,[])


if __name__=='__main__':unittest.main()
