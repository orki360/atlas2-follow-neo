"""Current release gate: Update 9 scenarios plus unchanged regression contracts.

Tests for Update 1-8 terminal-on-loss behavior are historical specifications;
Update 9 replaces that contract with ContinuousTests/SearchTests. Keeping their
source allows reviewing the change instead of silently rewriting history.
"""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
SUITES=[
 'test_update9','test_manual','test_speed','test_focus_transition','test_control_status',
 'test_update4.GuiRegressionTests','test_update4.SmoothTests','test_update5.TrackerTests',
 'test_update8.SessionSchedulingTests',
]
METHODS={
 'test_follow_core.TrackingTests':[
  'test_duplicate_result_does_not_confirm_or_refresh','test_far_distractor_is_rejected',
  'test_prediction_keeps_size_and_measurement_time','test_controller_hard_stale_zeroes_intent_and_overlay',
  'test_axis_reversal_brakes','test_model_letterbox_coordinates_and_nms','test_spacing_reports_cpp_terminal_edge_once'],
 'test_update6.TimingAndNoiseTests':[
  'test_filter_does_not_depend_on_latency_or_display_polling','test_prediction_reads_never_change_measurement_evidence',
  'test_stationary_jitter_does_not_create_large_velocity','test_constant_motion_stays_responsive_despite_box_noise',
  'test_size_change_does_not_invent_center_motion','test_inconsistent_boxes_disallow_extended_prediction',
  'test_weak_boxes_do_not_grant_extended_forward'],
 'test_update8.CadenceTests':[
  'test_sixty_source_frames_sample_to_thirty_without_cumulative_drift',
  'test_quantized_arrival_times_still_maintain_average_target','test_stalled_input_does_not_publish_a_catchup_burst',
  'test_receiver_skips_conversion_for_unsampled_frames','test_latest_value_wakes_consumer_and_does_not_build_backlog',
  'test_config_migration_is_once_and_preserves_tracking_values'],
 'test_v02.NewBehaviorTests':[
  'test_full_hd_clean_and_prediction_recordings_share_timestamps','test_gpu_unavailable_is_explicit_and_auto_falls_back',
  'test_live_box_at_display_frame_not_control_now','test_native_resolution_tracker_is_scale_equivalent',
  'test_recording_resolution_change_finalizes_existing_video','test_snapshot_metadata_stays_bound_to_analyzed_result'],
}
if __name__=='__main__':
    names=SUITES+[f'{cls}.{method}' for cls,methods in METHODS.items() for method in methods]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    sys.exit(0 if result.wasSuccessful() else 1)
