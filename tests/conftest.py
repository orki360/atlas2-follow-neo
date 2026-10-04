"""Keep pytest temporary files isolated from inaccessible shared Windows temp."""
from pathlib import Path
import uuid


def pytest_configure(config):
    if config.option.basetemp is None:
        project = Path(__file__).resolve().parents[1]
        target = (project / 'outputs' / 'test-tmp' / uuid.uuid4().hex).resolve()
        target.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(target)
