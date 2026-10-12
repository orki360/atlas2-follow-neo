"""Configurable SCAN cap, live application and explicit UI confirmation."""
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from follow_neo.types import Settings, settings_from_config
from follow_neo.search_test import SearchTest
from follow_neo.recovery_search import scan_command
from follow_neo.session import FollowSession
from follow_neo.manual import ManualControl
from follow_neo.gui import FollowLabWindow


def heading(t):
    return dict(time=t,yaw_deg=0.,source='AircraftAttitude')


class ScanSpeedSettingTests(unittest.TestCase):
    def test_settings_default_roundtrip_and_invalid_values(self):
        self.assertEqual(settings_from_config({}).scan_yaw_limit,.5)
        self.assertEqual(settings_from_config({'settings':{'scan_yaw_limit':.35}}).scan_yaw_limit,.35)
        for value in (-.01,1.01,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                Settings(scan_yaw_limit=value).validate()

    def test_live_scan_reduction_zero_and_increase_in_both_directions(self):
        for scenario,sign in [('LEFT',-1),('RIGHT',1)]:
            test=SearchTest(Settings(search_yaw_degrees=360,scan_yaw_limit=.8),heading(10),10,scenario,Mock())
            for i in range(1,21):
                t=10+i*.1;command=test.update(heading(t),t)[0]
            self.assertAlmostEqual(command,sign*.8)
            deadline=test.search.until;anchor=test.search.started
            for cap in (.2,0.,.65):
                test.settings=replace(test.settings,scan_yaw_limit=cap)
                for i in range(1,11):
                    t+=.1;command=test.update(heading(t),t)[0]
                    self.assertLessEqual(abs(command),cap+1e-9)
                self.assertAlmostEqual(command,sign*cap)
                self.assertEqual(test.search.until,deadline)
                self.assertEqual(test.search.started,anchor)

    def test_gate_validates_cap_and_boost_remains_full_scale(self):
        test=SearchTest(Settings(search_yaw_degrees=180),heading(10),10,'RIGHT',Mock())
        test.update(heading(10.3),10.3)
        search=test.decision['edge_search'];search['scan_yaw']=.9
        for cap in (0.,.2,.8):
            search['scan_yaw_limit']=cap
            self.assertEqual(scan_command(test.decision,10.3,10),cap)
        for cap in (-1,2,float('nan'),None,'50'):
            search['scan_yaw_limit']=cap
            self.assertEqual(scan_command(test.decision,10.3,10),0)
        boost=SearchTest(Settings(search_yaw_degrees=180,scan_yaw_limit=.2),heading(10),10,'RIGHT',Mock(),boost=True)
        self.assertEqual(boost.update(heading(10.1),10.1)[0],1.)

    def test_live_configuration_does_not_reset_session_or_search_test(self):
        session=object.__new__(FollowSession)
        session.settings=Settings();session.settings_lock=threading.Lock();session.log=Mock()
        session.generation=9;session.restart_control=Mock()
        session.configure_scan_speed(.3)
        self.assertEqual(session.settings,replace(Settings(),scan_yaw_limit=.3))
        self.assertEqual(session.generation,9);session.restart_control.assert_not_called()
        manual=ManualControl('localhost')
        test=SearchTest(Settings(),heading(10),10,'LEFT',Mock())
        manual.search_test=test
        manual.configure_scan_speed(.3)
        self.assertIs(manual.search_test,test)
        self.assertEqual(test.settings.scan_yaw_limit,.3)

    def test_apply_saves_only_confirmed_value_and_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            owner=object.__new__(FollowLabWindow)
            owner.settings=Settings();owner.scan_speed_var=Mock();owner.scan_speed_note=Mock()
            owner.notice=Mock();owner.canvas=Mock();owner.mark_dirty=Mock()
            owner.session=Mock();owner.manual=Mock()
            def save():
                path.write_text(json.dumps({'settings':{'scan_yaw_limit':owner.settings.scan_yaw_limit}}))
            owner.save_config=save
            owner.scan_speed_var.get.return_value='35'
            self.assertEqual(owner.settings.scan_yaw_limit,.5)
            owner.apply_scan_speed()
            self.assertEqual(settings_from_config(json.loads(path.read_text())).scan_yaw_limit,.35)
            owner.session.configure_scan_speed.assert_called_once_with(.35)
            owner.manual.configure_scan_speed.assert_called_once_with(.35)
            owner.scan_speed_var.get.return_value='101'
            with patch('follow_neo.gui.messagebox.showerror') as error:
                owner.apply_scan_speed();error.assert_called_once()
            self.assertEqual(owner.settings.scan_yaw_limit,.35)
            self.assertEqual(json.loads(path.read_text())['settings']['scan_yaw_limit'],.35)
            owner.scan_speed_var.get.return_value='20'
            owner.save_config=Mock(side_effect=OSError('save failed'))
            with patch('follow_neo.gui.messagebox.showerror') as error:
                owner.apply_scan_speed();error.assert_called_once()
            self.assertEqual(owner.settings.scan_yaw_limit,.35)
            owner.session.configure_scan_speed.assert_called_once()


if __name__=='__main__':unittest.main()
