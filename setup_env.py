"""Private Python 3.12 environment; selects OS-specific GPU support, no CUDA."""
from pathlib import Path
import json
import platform
import subprocess
import sys
import venv

root=Path(__file__).resolve().parent
if sys.version_info[:2]!=(3,12): raise SystemExit('Use Python 3.12: py -3.12 setup_env.py / python3.12 setup_env.py')
if platform.system()=='Darwin' and tuple(map(int,platform.mac_ver()[0].split('.')[:1]))<(13,):
    raise SystemExit('These pinned Python wheels require macOS 13 or later.')
env=root/'.venv'; py=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
if not py.exists(): venv.EnvBuilder(with_pip=True).create(env)
subprocess.run([str(py),'-c','import tkinter; print("Tk available:", tkinter.TkVersion)'],check=True)
desired='onnxruntime-directml' if sys.platform=='win32' else 'onnxruntime'
runtime_version='1.20.1' if sys.platform=='win32' else '1.22.1'
installed=json.loads(subprocess.check_output([str(py),'-c',
    'import importlib.metadata as m,json; print(json.dumps([x.metadata["Name"].lower() for x in m.distributions()]))'],text=True))
conflicts=[name for name in ('onnxruntime','onnxruntime-gpu','onnxruntime-directml') if name in installed and name!=desired]
if conflicts:
    subprocess.run([str(py),'-m','pip','uninstall','-y',*conflicts],check=True)
    # Distributions share an import directory; repair desired files if another was removed.
    subprocess.run([str(py),'-m','pip','install','--only-binary=:all:','--force-reinstall','--no-deps',desired+'=='+runtime_version],check=True)
subprocess.run([str(py),'-m','pip','install','--only-binary=:all:','-r',str(root/'requirements.txt')],check=True)
subprocess.run([str(py),'-m','pip','check'],check=True)
subprocess.run([str(py),str(root/'run_lab.py'),'--check','--compute','Auto'],check=True)
print('SETUP PASSED. Read compute status above. Run RUN_FOLLOW_NEO.cmd or bash RUN_FOLLOW_NEO.sh')
