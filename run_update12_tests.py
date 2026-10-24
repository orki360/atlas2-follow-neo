"""Existing tracking/TCP release gate, also used by the process-isolation build."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from run_update9_tests import SUITES, METHODS

if __name__ == '__main__':
    names = ['test_recovery_cadence', 'test_recovery_diagnostics', 'test_dual_class', 'test_approach_revision13', 'test_search_360', 'test_search_motion', 'test_boost_scan_test', 'test_state_diagram', 'test_control_pause', 'test_search_debug', 'test_search_direction', 'test_single_target', 'test_stabilization', 'test_continuity', 'test_control_scan_fix', 'test_update12', 'test_update10'] + SUITES + [
        f'{cls}.{method}' for cls, methods in METHODS.items() for method in methods]
    names.insert(0, 'test_model_resolution')
    names.insert(0, 'test_delayed_overlay')
    names.insert(0, 'test_nano_bridge')
    names[:0] = ['test_tracking_recovery8', 'test_tracking_recovery7', 'test_link_recovery', 'test_nano_handoff', 'test_visual_continuity', 'test_heading_stream']
    names.insert(0, 'test_tracking_recovery9')
    names.insert(0, 'test_tracking_recovery10')
    names.insert(0, 'test_association_sampling')
    names.insert(0, 'test_motion_continuity')
    names.insert(0, 'test_scan_controls')
    names.insert(0, 'test_control_resync')
    names.insert(0, 'test_recording_budget')
    names.insert(0, 'test_runtime_mode')
    names.insert(0, 'test_gmc_settings')
    names[:0] = ['test_position_handoff', 'test_nano_position', 'test_early_curve', 'test_diagnostic_epoch']
    names.insert(0, 'test_send_cadence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    raise SystemExit(0 if result.wasSuccessful() else 1)
