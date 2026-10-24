"""Inspect actual Tk widgets without connecting to a phone or aircraft."""
from pathlib import Path
import sys
import tempfile
import json
import tkinter as tk

sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from VERSION import VERSION,REVISION
from follow_neo.ui.window import FollowLabWindow
from follow_neo.perception.models import DEFAULT_MODEL,MODEL_NAMES,DUAL_MODELS,model_from_config
from follow_neo.shared.types import Settings,settings_from_config

with tempfile.TemporaryDirectory() as folder:
    root=tk.Tk()
    root.withdraw()
    try:
        window=FollowLabWindow(root,Path(folder))
        root.update_idletasks()
        assert VERSION=='18.0.1'
        assert window.hard_button.cget('text')=='REAL-TIME HARD'
        assert window.tcp_send_hz==20
        assert tuple(window.send_rate_picker.cget('values'))==('20','25','30')
        assert '50.00 ms' in window.send_rate_note.get()
        assert root.title()==f'ATLAS2 Follow NEO | Version {VERSION} / {REVISION} - BoT-SORT'
        assert window.version_label.cget('text')=='v'+VERSION
        assert window.revision_label.cget('text')==REVISION
        assert window.model.get()==DEFAULT_MODEL
        assert tuple(window.model_picker.cget('values'))==MODEL_NAMES
        for name in DUAL_MODELS:
            window.model.set(name)
            window.save_config()
            saved=json.loads(window.config_path.read_text(encoding='utf-8'))
            assert model_from_config(saved)==name
            assert window.session is None and window.manual is None
        window.model.set(DEFAULT_MODEL)
        assert window.tabs.tab(window.diagram_tab,'text')=='Live states'
        assert window.tabs.select()==str(window.diagram_tab)
        assert window.search_auto.get() is True
        assert window.nano_mode.get()==Settings().nano_mode
        assert window.tabs.tab(window.botsort_tab,'text')=='BoT-SORT'
        assert window.gmc_resolution.get()=='480x270'
        assert window.vars['gmc_max_corners'].get()=='400'
        assert tuple(window.gmc_resolution_picker.cget('values'))==('480x270','640x360','960x540','1280x720','1920x1080')
        before_gmc=window.settings
        window.vars['confidence'].set('0.65')
        window.gmc_resolution.set('480x270');window.vars['gmc_max_corners'].set('300')
        window.apply_gmc()
        restored=settings_from_config(json.loads(window.config_path.read_text()))
        assert restored.gmc_resolution=='480x270' and restored.gmc_max_corners==300
        assert restored.confidence==before_gmc.confidence
        assert 'Changes pending' in window.settings_note.get()
        window.vars['confidence'].set(str(before_gmc.confidence))
        window.gmc_resolution.set('640x360');window.vars['gmc_max_corners'].set('400');window.apply_gmc()
        for key,value in (('botsort_match_threshold',.85),('botsort_low_match_threshold',.55),
                          ('botsort_unconfirmed_match_threshold',.75)):
            assert key in window.vars
            window.vars[key].set(str(value))
        window.apply()
        restored=settings_from_config(json.loads(window.config_path.read_text()))
        assert (restored.botsort_match_threshold,restored.botsort_low_match_threshold,
                restored.botsort_unconfirmed_match_threshold)==(.85,.55,.75)
        for mode in ('off','shadow','guarded'):
            window.nano_mode.set(mode)
            window.settings=window.read_settings();window.save_config()
            assert json.loads(window.config_path.read_text())['settings']['nano_mode']==mode
        window.nano_mode.set('shadow');window.settings=window.read_settings()
        assert window.session is None and window.manual is None
        config_before=window.config_path.read_bytes()
        settings_before=window.settings
        window.toggle_real_time();root.update_idletasks()
        assert window.real_time and window.realtime_button.cget('text')=='Real-Time: ON'
        assert str(window.capture_button.cget('state'))=='disabled'
        assert str(window.raw_record_button.cget('state'))=='disabled'
        assert not window.sidebar.winfo_manager() and not window.view_bar.winfo_manager()
        window.refresh()
        window.toggle_real_time();root.update_idletasks()
        assert not window.real_time and window.realtime_button.cget('text')=='Real-Time: OFF'
        assert window.sidebar.winfo_manager()=='grid' and window.view_bar.winfo_manager()=='pack'
        assert window.settings is settings_before and window.config_path.read_bytes()==config_before
        assert window.session is None and window.manual is None
        assert window.tabs.tab(window.motion_tab,'text')=='Motion'
        assert len(window.motion_controls)==15
        assert tuple(window.scan_observation_picker.cget('values'))==('stop','slowdown')
        assert window.scan_observation_mode.get()=='stop'
        window.vars['scan_observation_degrees'].set('80')
        window.vars['scan_yaw_limit'].set('45')
        window.vars['scan_observation_yaw'].set('6')
        window.scan_observation_mode.set('slowdown');window.apply()
        restored=settings_from_config(json.loads(window.config_path.read_text()))
        assert restored.scan_observation_degrees==80 and restored.scan_yaw_limit==.45
        assert restored.scan_observation_yaw==.06 and restored.scan_observation_mode=='slowdown'
        window.vars['scan_observation_degrees'].set('360')
        window.vars['scan_observation_yaw'].set('60');window.apply()
        restored=settings_from_config(json.loads(window.config_path.read_text()))
        assert restored.scan_observation_degrees==360 and restored.scan_observation_yaw==.6
        window.vars['scan_observation_degrees'].set('720');window.apply()
        assert settings_from_config(json.loads(window.config_path.read_text())).scan_observation_degrees==720
        assert window.coordinated_turns.get() is True
        window.coordinated_turns.set(False)
        window.vars['forward_window_width'].set('65')
        window.vars['forward_window_height'].set('35')
        window.vars['forward_yaw_rate_limit'].set('50')
        window.vars['search_look_seconds'].set('0.20')
        window.forward_require_nano.set(True)
        window.apply()
        saved=json.loads(window.config_path.read_text())
        restored=settings_from_config(saved)
        assert restored.forward_window_width==.65 and restored.forward_window_height==.35
        assert restored.forward_yaw_rate_limit==50 and restored.search_look_seconds==.20
        assert restored.forward_require_nano is True
        assert restored.coordinated_turns is False
        window.coordinated_turns.set(True);window.apply()
        assert settings_from_config(json.loads(window.config_path.read_text())).coordinated_turns is True
        assert window.session is None and window.manual is None
        assert callable(window.key_state_reader)
        texts=[]
        def inspect(widget):
            try:texts.append(str(widget.cget('text')))
            except tk.TclError:pass
            for child in widget.winfo_children():inspect(child)
        inspect(root)
        assert 'Search angle (deg)' in texts,texts
        assert 'START SEARCH TEST' in texts and 'STOP SEARCH TEST' in texts
        assert window.search_test_scenario.get()=='CENTER'
        window.start_search_test()
        assert 'Connect' in window.notice.get()
        assert 'Apply / keep target' in texts
        assert 'Loss grace (seconds)' in texts
        assert 'Auto scan time for angle (up to 35 s)' in texts
        assert float(window.vars['search_grace_seconds'].get())==.45
        panel=window.state_diagram
        before=len(panel.canvas.find_all())
        for i,state in enumerate(('TRACK','DIRECTIONAL_SEARCH','SEARCH_PAUSED','REACQUIRE','HOVER_WAIT','WAIT_VIDEO','ERROR')):
            stamp=10.+i
            d=dict(state=state,prediction_time=stamp,reason='offline_test',spacing_phase='APPROACH',
                   edge_search=dict(phase='BOOST' if i==1 else 'SCAN',paused=i==2))
            control=dict(connected=True,active=True,mode='DANCE',time=stamp,paused=i==2)
            panel.update_state(d,control,True,True,stamp)
            assert panel.last_view['state']==state
            assert panel.canvas.itemcget(panel.boxes[state],'fill')!= '#243746'
        assert len(panel.canvas.find_all())==before
        print(f'PASS: offline Tk GUI {VERSION} / {REVISION}; dual-class model is default; all state highlights update; offline start refused; no connections.')
    finally:
        root.destroy()
