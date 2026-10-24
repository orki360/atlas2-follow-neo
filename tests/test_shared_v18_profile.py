import json
from pathlib import Path
import tempfile
import unittest

from load_v18_profile import PROFILE, load_profile, profile_config
from follow_neo.shared.types import settings_from_config


class SharedV18ProfileTests(unittest.TestCase):
    def test_snapshot_and_runtime_validation(self):
        config = profile_config(PROFILE)
        settings = settings_from_config(config)
        self.assertEqual(config['tcp_send_hz'], 25)
        self.assertEqual(config['axis_limits'], dict(yaw=1.0, vertical=0.0, roll=0.353, forward=0.06))
        self.assertEqual((settings.gmc_resolution, settings.gmc_max_corners), ('480x270', 400))
        self.assertTrue(settings.nano_primary_position)
        self.assertTrue({'phone_ip', 'compute', 'gpu_adapter'}.isdisjoint(config))

    def test_preserves_machine_settings_and_backs_up_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            original = b'{"phone_ip":"10.0.0.5","compute":"CPU","gpu_adapter":2,"tcp_send_hz":20}\r\n'
            path.write_bytes(original)
            merged, backup = load_profile(PROFILE, path)
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(json.loads(path.read_text()), merged)
            self.assertEqual((merged['phone_ip'], merged['compute'], merged['gpu_adapter']), ('10.0.0.5', 'CPU', 2))
            self.assertEqual(merged['tcp_send_hz'], 25)
            repeated, second_backup = load_profile(PROFILE, path)
            self.assertEqual(repeated, merged)
            self.assertNotEqual(backup, second_backup)

    def test_new_install_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            preview, backup = load_profile(PROFILE, path, dry_run=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            actual, backup = load_profile(PROFILE, path)
            self.assertEqual(actual, preview)
            self.assertIsNone(backup)
            settings_from_config(actual)

    def test_invalid_existing_file_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            path.write_bytes(b'not json')
            with self.assertRaises(ValueError):
                load_profile(PROFILE, path)
            self.assertEqual(path.read_bytes(), b'not json')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_profile_never_writes(self):
        for key, value in [('phone_ip', '10.0.0.1'), ('tcp_send_hz', 50),
                           ('axis_limits', dict(yaw=1, vertical=0, roll=0, forward=float('nan'))),
                           ('settings', {})]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                profile = json.loads(PROFILE.read_text(encoding='utf-8'))
                profile['config'][key] = value
                source = Path(directory) / 'profile.json'
                source.write_text(json.dumps(profile), encoding='utf-8')
                destination = Path(directory) / 'config.local.json'
                with self.assertRaises(ValueError):
                    load_profile(source, destination)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
