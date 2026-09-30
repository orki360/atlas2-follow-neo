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
        assert VERSION=='11.1'
        assert window.version_label.cget('text')=='v11.1'
        assert window.revision_label.cget('text')=='control-auto-resume-7'
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
        print('PASS: actual Tk GUI initializes; version 11.1 / control-auto-resume-7, search buttons and scenario; offline start refused; no connections.')
    finally:
        root.destroy()
