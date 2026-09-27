"""Current gate: all Update 9 regression contracts plus Update 10 scenarios."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from run_update9_tests import SUITES,METHODS
if __name__=='__main__':
    names=['test_update10']+SUITES+[f'{cls}.{method}' for cls,methods in METHODS.items() for method in methods]
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
