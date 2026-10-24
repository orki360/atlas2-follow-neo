"""Offline process-isolation gate: loopback TCP only, no phone required."""
import sys
import unittest
from pathlib import Path

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_process_runtime','test_session_shutdown','test_hard_runtime']))
    raise SystemExit(0 if result.wasSuccessful() else 1)
