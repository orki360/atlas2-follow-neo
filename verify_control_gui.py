"""Inspect actual Tk widgets without connecting to a phone or aircraft."""
from pathlib import Path
import sys
import tempfile
import tkinter as tk

sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from VERSION import VERSION
from follow_neo.gui import FollowLabWindow

with tempfile.TemporaryDirectory() as folder:
    root=tk.Tk()
    root.withdraw()
    try:
        window=FollowLabWindow(root,Path(folder))
        root.update_idletasks()
        assert VERSION=='12.0.0'
        assert window.version_label.cget('text')=='v12.0.0'
        assert window.revision_label.cget('text')=='live-state-diagram-8'
        assert window.tabs.tab(window.diagram_tab,'text')=='Live states'
        assert window.tabs.select()==str(window.diagram_tab)
        assert window.search_auto.get() is True
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
        assert 'Auto scan time for angle (max 10 s)' in texts
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
        print('PASS: actual Tk GUI initializes; version 12.0.0 / live-state-diagram-8; all state highlights update without recreating canvas items; offline start refused; no connections.')
    finally:
        root.destroy()
