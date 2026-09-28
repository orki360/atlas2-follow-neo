"""Release 11.0.0: existing release gate and real BoT-SORT integration contracts."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from run_update9_tests import SUITES, METHODS

if __name__ == '__main__':
    names = ['test_continuity', 'test_control_scan_fix', 'test_update12', 'test_update10'] + SUITES + [
        f'{cls}.{method}' for cls, methods in METHODS.items() for method in methods]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    raise SystemExit(0 if result.wasSuccessful() else 1)
