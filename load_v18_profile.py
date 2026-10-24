"""Offline, backed-up import of the shared v18 experiment settings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

from follow_neo.perception.models import MODEL_NAMES
from follow_neo.shared.types import Settings, settings_from_config
from follow_neo.transport.cadence import checked_send_hz

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / 'profiles' / 'v18_shared.json'
PORTABLE_KEYS = {
    'schema_version', 'model_name', 'recording_profile', 'codec',
    'settings', 'axis_limits', 'tcp_send_hz',
}


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'{Path(path).name}: expected a JSON object')
    return value


def profile_config(path):
    profile = read_object(path)
    if profile.get('profile_format') != 1 or profile.get('application_version') != '18.0.1':
        raise ValueError('Expected the v18.0.1 shared profile (format 1)')
    config = profile.get('config')
    if not isinstance(config, dict) or set(config) != PORTABLE_KEYS:
        raise ValueError('Profile must contain exactly the portable configuration keys')
    if config['schema_version'] != 8 or config['model_name'] not in MODEL_NAMES:
        raise ValueError('Unsupported configuration schema or model')
    if config['codec'] != 'h264' or config['recording_profile'] != 'Auto':
        raise ValueError('Expected the shared h264 / Auto recording settings')
    settings = config['settings']
    defaults = asdict(Settings())
    if not isinstance(settings, dict) or set(settings) != set(defaults):
        raise ValueError('Profile must contain all v18 tracking settings')
    for key, default in defaults.items():
        value = settings[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, str):
            valid = isinstance(value, str)
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f'Invalid setting type: {key}')
    settings_from_config(config)
    checked_send_hz(config['tcp_send_hz'])
    axes = config['axis_limits']
    if not isinstance(axes, dict) or set(axes) != {'yaw', 'vertical', 'roll', 'forward'}:
        raise ValueError('Expected all four axis limits')
    for key, value in axes.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid axis limit: {key}')
    return config


def load_profile(profile_path, config_path, *, dry_run=False):
    """Preserve machine settings; back up exact bytes before atomic replacement."""
    portable = profile_config(profile_path)
    config_path = Path(config_path)
    existing = read_object(config_path) if config_path.exists() else {}
    merged = {**existing, **portable}
    payload = json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if dry_run:
        return merged, None
    backup = None
    if config_path.exists():
        backup_root = config_path.parent / 'backups'
        backup_root.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        backup_dir = Path(tempfile.mkdtemp(prefix=f'before_v18_profile_{stamp}_', dir=backup_root))
        backup = backup_dir / 'config.local.json'
        backup.write_bytes(config_path.read_bytes())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=config_path.parent,
                                         prefix='.v18_profile_', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(config_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return merged, backup


def main():
    parser = argparse.ArgumentParser(description='Load the shared v18 settings. Close Follow NEO first.')
    parser.add_argument('--dry-run', action='store_true', help='Validate without writing any files')
    args = parser.parse_args()
    try:
        config, backup = load_profile(PROFILE, ROOT / 'config.local.json', dry_run=args.dry_run)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f'Profile not loaded: {error}\n')
    print('Validated only; no files changed.' if args.dry_run else 'Shared v18 profile loaded.')
    print(f"TCP: {config['tcp_send_hz']} Hz; model: {config['model_name']}")
    print('Axis limits (%): ' + ', '.join(f'{key}={value * 100:g}' for key, value in config['axis_limits'].items()))
    print(f"GMC: {config['settings']['gmc_resolution']}, {config['settings']['gmc_max_corners']} corners")
    print('Phone IP, compute backend and GPU adapter are preserved when present.')
    print('Check those local fields in the GUI before connecting. This command does not connect or start Dance.')
    if backup:
        print(f'Previous settings backup: {backup}')


if __name__ == '__main__':
    main()
