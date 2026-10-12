"""ROI settings for the main Analyzed frame display (Tk main thread only)."""
import tkinter as tk
from tkinter import ttk, messagebox
from .roi import RoiSettings


class RoiPanel:
    def __init__(self, owner, config):
        self.owner=owner
        self.settings=RoiSettings.from_config(config)
        owner.roi_settings=self.settings
        self.tab=ttk.Frame(owner.tabs,padding=8)
        owner.tabs.add(self.tab,text='ROI')
        self.enabled=tk.BooleanVar(value=self.settings.enabled)
        ttk.Checkbutton(self.tab,text='Enable ROI classification',variable=self.enabled).pack(anchor='w')
        ttk.Label(self.tab,text='NEO candidates only. Crops run sequentially.\nEntry limits are original-frame pixels.\nOversized boxes use normal detection.',wraplength=370).pack(anchor='w',pady=6)
        self.variables={}
        fields=[('Crop width','width'),('Crop height','height'),('Max BBOX width','max_box_width'),
                ('Max BBOX height','max_box_height'),('Confirmations','confirmations'),
                ('ROI confidence','confidence'),('Matching IoU','match_iou'),('Max confirmation gap (s)','max_gap_seconds'),
                ('Confirmed recheck interval (s)','recheck_seconds')]
        for label,name in fields:
            row=ttk.Frame(self.tab);row.pack(fill='x',pady=2)
            ttk.Label(row,text=label).pack(side='left')
            var=tk.StringVar(value=str(getattr(self.settings,name)));self.variables[name]=var
            ttk.Entry(row,textvariable=var,width=9).pack(side='right')
        ttk.Button(self.tab,text='Apply ROI settings',command=self.apply).pack(fill='x',pady=6)
        ttk.Button(self.tab,text='Show ROI in Analyzed frame',command=self.show_analyzed).pack(fill='x')
        self.note=tk.StringVar(value='Two confirmations use distinct processed frames.\nFresh tracking: periodic ROI checks.\nAcquisition / loss: every frame. 0 s = every frame.')
        ttk.Label(self.tab,textvariable=self.note,wraplength=370).pack(anchor='w',pady=8)
        ttk.Label(self.tab,text='Full frame: NEO yellow / MINI red\nFrom ROI: NEO turquoise / MINI orange\nROI boundary: purple dashed',wraplength=370).pack(anchor='w',pady=4)
        self.status=tk.StringVar(value='ROI drawings appear in Analyzed frame.')
        ttk.Label(self.tab,textvariable=self.status,wraplength=370).pack(anchor='w',pady=4)

    def read_settings(self):
        values={'enabled':self.enabled.get()}
        for name,var in self.variables.items():
            values[name]=float(var.get()) if name in ('confidence','match_iou','max_gap_seconds','recheck_seconds') else int(var.get())
        return RoiSettings(**values).validate()

    def apply(self):
        try:
            settings=self.read_settings()
            if self.owner.session:self.owner.session.configure_roi(settings)
            self.settings=settings;self.owner.roi_settings=settings
            self.owner.save_config()
            self.note.set('ROI settings applied / saved. Confirmation restarts.')
        except (ValueError,OSError) as exc:messagebox.showerror('ROI settings',str(exc),parent=self.owner.root)

    def show_analyzed(self):
        self.owner.view.set('Analyzed frame')
        self.owner.draw_key=None

    def update(self,result):
        if result is None:
            self.status.set('ROI drawings appear in Analyzed frame. Waiting for video.')
            return
        self.status.set(f"Frame {result['frame'].frame_id} | ROIs {result.get('roi_inferences',0)}"
                        f" | cached {result.get('roi_cached',0)} | ROI {result.get('roi_ms',0):.1f} ms | {result.get('roi_note','waiting')}")
