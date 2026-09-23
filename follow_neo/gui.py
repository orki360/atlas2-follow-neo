"""Tkinter GUI. All Tk calls stay on the main thread."""
from dataclasses import asdict, fields
from pathlib import Path
from collections import deque
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from .types import Settings, settings_from_config
from .session import FollowSession
from .overlay import render_bundle
from .manual import ManualControl, MOVEMENT_KEYS, AXES, DEFAULT_AXIS_LIMITS
from .control_status import control_indicator


# Windows keeps these virtual-key codes stable even when the active keyboard
# layout produces Hebrew characters. Tk's keysym changes with the layout.
WINDOWS_PHYSICAL_KEYS={65:'a',68:'d',69:'e',70:'f',81:'q',82:'r',83:'s',87:'w',88:'x'}


def normalized_event_key(event,platform=None):
    key=str(getattr(event,'keysym','')).lower()
    platform=sys.platform if platform is None else platform
    keycode=getattr(event,'keycode',None)
    if platform=='win32' and isinstance(keycode,int):
        return WINDOWS_PHYSICAL_KEYS.get(keycode,key)
    return key


def safe_focus_widget(root):
    """A ttk popdown exists in Tcl but is not a registered Python widget."""
    try: return root.focus_get()
    except KeyError:
        try:
            name=str(root.tk.call('focus'))
            while name:
                try: return root.nametowidget(name)
                except KeyError: name=name.rsplit('.',1)[0]
        except tk.TclError: pass
    except tk.TclError: pass
    return None


class FollowLabWindow:
    def __init__(self,root,base):
        self.root=root; self.base=Path(base); self.session=None; self.stopping=False
        self.manual=None; self.pressed=set(); self.key_releases={}
        self.dance_rejection=''; self.last_control_indicator=None
        self.dance_selected=False; self.dance_terminal_seen=False
        self.closing=False; self.focus_check_id=None
        self.last_image=None; self.last_frame=None; self.last_meta=None; self.photo=None; self.draw_key=None
        self.rates=deque(maxlen=90); self.started=time.monotonic()
        self.config_path=self.base/'config.local.json'; cfg={}
        if self.config_path.exists():
            try: cfg=json.loads(self.config_path.read_text(encoding='utf-8'))
            except (OSError,ValueError): pass
        self.root.title('ATLAS2 Follow NEO | Live Perception Lab v0.11.0 - Update 9')
        self.root.geometry('1440x880'); self.root.minsize(1200,760)
        self.root.configure(bg='#101820')
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        style=ttk.Style(); style.theme_use('clam')
        style.configure('.',background='#17232d',foreground='#e2edf4',font=('Segoe UI',10))
        style.configure('TButton',padding=(10,7),background='#2d4658')
        style.map('TButton',background=[('active','#426780')])
        style.configure('TEntry',fieldbackground='#233543',foreground='white',insertcolor='white')
        style.configure('TCombobox',fieldbackground='#233543',foreground='white')
        style.map('TCombobox',fieldbackground=[('readonly','#233543')],foreground=[('readonly','#e2edf4')])
        style.configure('TSpinbox',fieldbackground='#233543',foreground='white',arrowsize=14)
        style.configure('TLabelframe.Label',foreground='#6ad5cb',font=('Segoe UI',10,'bold'))
        style.configure('Header.TLabel',font=('Segoe UI',18,'bold'),background='#101820')
        top=ttk.Frame(root,padding=12); top.pack(fill='x')
        ttk.Label(top,text='ATLAS2  /  FOLLOW NEO',style='Header.TLabel').pack(side='left')
        ttk.Label(top,text='NEO DANCE + MANUAL KEYBOARD',foreground='#6ad5cb').pack(side='right')
        bar=ttk.Frame(root,padding=(12,4)); bar.pack(fill='x')
        ttk.Label(bar,text='Phone IP').pack(side='left')
        self.ip=tk.StringVar(value=cfg.get('phone_ip',''))
        ttk.Entry(bar,textvariable=self.ip,width=19).pack(side='left',padx=8)
        ttk.Label(bar,text='Video: 9999').pack(side='left',padx=5)
        self.codec=tk.StringVar(value=cfg.get('codec','h264'))
        ttk.Combobox(bar,textvariable=self.codec,values=['h264','h265'],state='readonly',width=7).pack(side='left',padx=5)
        self.connect_button=ttk.Button(bar,text='Connect',command=self.connect); self.connect_button.pack(side='left',padx=4)
        self.disconnect_button=ttk.Button(bar,text='Disconnect',command=self.disconnect,state='disabled'); self.disconnect_button.pack(side='left',padx=4)
        self.record=tk.BooleanVar(value=False)
        ttk.Checkbutton(bar,text='Save raw video for this session',variable=self.record).pack(side='left',padx=12)

        compute_bar=ttk.Frame(root,padding=(12,4)); compute_bar.pack(fill='x')
        ttk.Label(compute_bar,text='Compute').pack(side='left')
        self.compute=tk.StringVar(value=cfg.get('compute','Auto'))
        self.compute_picker=ttk.Combobox(compute_bar,textvariable=self.compute,values=['Auto','GPU','CPU'],state='readonly',width=9)
        self.compute_picker.pack(side='left',padx=8)
        ttk.Label(compute_bar,text='GPU adapter (-1 = auto)').pack(side='left')
        self.device=tk.IntVar(value=cfg.get('gpu_adapter',-1))
        self.device_picker=ttk.Spinbox(compute_bar,textvariable=self.device,from_=-1,to=31,width=4)
        self.device_picker.pack(side='left',padx=8)
        ttk.Label(compute_bar,text='YOLO + Kalman | short prediction is marked separately',foreground='#8cbbb8').pack(side='left',padx=12)

        manual_bar=ttk.Frame(root,padding=(12,4)); manual_bar.pack(fill='x')
        self.control_bar=manual_bar
        self.manual_button=ttk.Button(manual_bar,text='Connect keyboard',command=self.toggle_manual)
        self.manual_button.pack(side='left')
        ttk.Button(manual_bar,text='Enable (E)',command=self.enable_manual).pack(side='left',padx=4)
        ttk.Button(manual_bar,text='Release (Q / Esc)',command=self.release_manual).pack(side='left',padx=4)
        ttk.Button(manual_bar,text='Takeoff (F)',command=lambda:self.manual_action('takeoff')).pack(side='left',padx=4)
        ttk.Button(manual_bar,text='Land (R)',command=lambda:self.manual_action('land')).pack(side='left',padx=4)
        self.dance_button=ttk.Button(manual_bar,text='START DANCE',command=self.toggle_dance)
        self.dance_button.pack(side='left',padx=8)
        self.speed_bar=ttk.Frame(root,padding=(12,4)); self.speed_bar.pack(fill='x')
        ttk.Label(self.speed_bar,text='NORMAL AXIS LIMITS',foreground='#6ad5cb').pack(side='left',padx=(0,8))
        stored_limits=cfg.get('axis_limits',{})
        self.axis_limit_vars={}; self.axis_limit_labels={}; self.axis_entry_vars={}
        axis_labels=(('Up / Down','vertical'),('Yaw L / R','yaw'),
                     ('Forward / Back','forward'),('Side L / R','roll'))
        for label,axis in axis_labels:
            group=ttk.Frame(self.speed_bar); group.pack(side='left',padx=5)
            value=max(0.,min(1.,float(stored_limits.get(axis,DEFAULT_AXIS_LIMITS[axis]))))*100
            self.axis_limit_vars[axis]=tk.DoubleVar(value=value)
            self.axis_entry_vars[axis]=tk.StringVar(value=f'{value:g}')
            self.axis_limit_labels[axis]=tk.StringVar(value=f'{value:.1f}%')
            ttk.Label(group,text=label).pack(anchor='w')
            row=ttk.Frame(group); row.pack()
            ttk.Scale(row,from_=0,to=100,variable=self.axis_limit_vars[axis],length=105,
                      command=lambda new_value,selected=axis:self.change_axis_speed(selected,new_value)).pack(side='left')
            exact=ttk.Spinbox(row,textvariable=self.axis_entry_vars[axis],from_=0,to=100,
                              increment=.5,width=6,
                              command=lambda selected=axis:self.change_axis_speed(
                                  selected,self.axis_entry_vars[selected].get()))
            exact.pack(side='left',padx=(4,0))
            exact.bind('<Return>',lambda event,selected=axis:self.change_axis_speed(
                selected,self.axis_entry_vars[selected].get()))
            exact.bind('<FocusOut>',lambda event,selected=axis:self.change_axis_speed(
                selected,self.axis_entry_vars[selected].get()))
            ttk.Label(row,textvariable=self.axis_limit_labels[axis],width=7).pack(side='left',padx=(3,0))
        ttk.Label(self.speed_bar,text='Normal caps\nSearch: yaw 100%',foreground='#8cbbb8').pack(side='left',padx=8)
        self.manual_status=tk.StringVar(value='Keyboard disconnected | Start Control Server on phone')
        ttk.Label(root,textvariable=self.manual_status,padding=(12,2),foreground='#6ad5cb').pack(fill='x')
        self.dance_status=tk.StringVar(value='Dance off | Take off manually, enable control, then Start NEO Dance')
        self.control_banner=tk.Label(root,text='CONTROL DISCONNECTED',font=('Segoe UI',19,'bold'),
                                     bg='#354452',fg='white',pady=8)
        self.control_banner.pack(fill='x',padx=12,pady=(4,0))
        ttk.Label(root,textvariable=self.dance_status,padding=(12,4),foreground='#d9b56c').pack(fill='x')
        search_values=asdict(settings_from_config(cfg))
        self.search_vars={key:tk.StringVar(value=f'{search_values[key]:g}')
                          for key in ('search_yaw_degrees','edge_search_seconds')}
        self.search_bar=ttk.Frame(root,padding=(12,4));self.search_bar.pack(fill='x')
        for label,key,lo,hi,step in (('Search YAW (deg)','search_yaw_degrees',0,180,5),
                                     ('Search time (s)','edge_search_seconds',2,10,.5)):
            ttk.Label(self.search_bar,text=label).pack(side='left',padx=(0,8))
            ttk.Spinbox(self.search_bar,textvariable=self.search_vars[key],from_=lo,to=hi,
                        increment=step,width=7).pack(side='left',padx=(0,18))
        ttk.Button(self.search_bar,text='Apply search',command=self.apply).pack(side='left',padx=4)
        ttk.Label(self.search_bar,text='0 deg = no turn | angle OR time limit | heading required',
                  foreground='#8cbbb8').pack(side='left',padx=12)
        ttk.Label(root,text='W/S: up/down   A/D: yaw   Arrows: forward/back/left/right   Measured yaw search overrides only yaw to 100%',padding=(12,2)).pack(fill='x')
        self.root.bind('<KeyPress>',self.key_press)
        self.root.bind('<KeyRelease>',self.key_release)
        self.root.bind('<FocusOut>',self.focus_out)
        body=ttk.Frame(root,padding=12); body.pack(fill='both',expand=True)
        body.columnconfigure(0,weight=1); body.rowconfigure(0,weight=1)
        left=ttk.Frame(body); left.grid(row=0,column=0,sticky='nsew',padx=(0,12))
        viewbar=ttk.Frame(left); viewbar.pack(fill='x',pady=(0,8))
        self.view_bar=viewbar
        ttk.Label(viewbar,text='Display').pack(side='left')
        self.view=tk.StringVar(value='Analyzed frame')
        ttk.Combobox(viewbar,textvariable=self.view,values=['Analyzed frame','Live prediction'],state='readonly',width=19).pack(side='left',padx=8)
        capture=ttk.Frame(viewbar); capture.pack(side='right')
        self.capture_mode=tk.StringVar(value='Snapshot')
        ttk.Combobox(capture,textvariable=self.capture_mode,values=['Snapshot','Clean video','Prediction video','Both videos'],
                     state='readonly',width=18).pack(side='left',padx=6)
        self.capture_button=ttk.Button(capture,text='Save',command=self.capture_action,width=7)
        self.capture_button.pack(side='left')
        self.recording_status=tk.StringVar(value='No recording')
        ttk.Label(left,textvariable=self.recording_status,foreground='#d9b56c').pack(fill='x',pady=(0,5))
        self.canvas=tk.Canvas(left,bg='#090f15',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Button-1>',lambda event:self.canvas.focus_set())
        self.canvas.create_text(410,200,text='Connect to the phone video server\n\nMini 4 Pro > RC-N3 > MSDKRemote > this window',
                                fill='#93a9ba',font=('Segoe UI',15),justify='center',tags='hint')
        self.video_stats=tk.StringVar(value='Waiting for live video')
        ttk.Label(left,textvariable=self.video_stats,wraplength=950,padding=8).pack(fill='x')

        sidebar=ttk.Frame(body); sidebar.grid(row=0,column=1,sticky='nsew')
        sidecanvas=tk.Canvas(sidebar,width=320,bg='#17232d',highlightthickness=0)
        scroll=ttk.Scrollbar(sidebar,orient='vertical',command=sidecanvas.yview)
        sidecanvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y'); sidecanvas.pack(side='left',fill='both',expand=True)
        right=ttk.Frame(sidecanvas)
        sidecanvas.create_window((0,0),window=right,anchor='nw',width=314)
        right.bind('<Configure>',lambda event:sidecanvas.configure(scrollregion=sidecanvas.bbox('all')))
        self.state=tk.StringVar(value='DISCONNECTED')
        self.metrics=tk.StringVar(value='Model: best.onnx | waiting\nTracking intent is preview only.')
        self.intent=tk.StringVar(value='Yaw +0.000   Up +0.000\nSide +0.000  Forward +0.000')
        status=ttk.LabelFrame(right,text='Tracking status',padding=10); status.pack(fill='x')
        ttk.Label(status,textvariable=self.state,font=('Segoe UI',14,'bold'),foreground='#6ad5cb').pack(anchor='w')
        ttk.Label(status,textvariable=self.metrics,justify='left',wraplength=290).pack(anchor='w',pady=6)
        ttk.Label(status,text='Tracking intent (sent only in Dance)',foreground='#d9b56c').pack(anchor='w')
        ttk.Label(status,textvariable=self.intent,font=('Consolas',11)).pack(anchor='w',pady=5)
        settings=ttk.LabelFrame(right,text='Detection and tracking',padding=10); settings.pack(fill='x',pady=10)
        values=asdict(settings_from_config(cfg))
        self.loaded_values=values
        self.vars=dict(self.search_vars)
        controls=[('YOLO acquire confidence','confidence',.05,.95,.05),('New track confidence','new_track_confidence',.05,.99,.05),
                  ('NMS IoU','nms_iou',.05,.95,.05),('Video processing FPS','video_fps',1,60,1),
                  ('Inference target FPS','inference_fps',1,60,1),
                  ('Result stale (seconds)','stale_seconds',.1,2,.05),
                  ('Stop BBOX width %','stop_width',1,50,2),('Yaw limit %','yaw_limit',10,100,5),
                  ('Vertical limit %','vertical_limit',10,100,5),('Forward limit %','forward_limit',0,100,5)]
        self.percent={'stop_width','yaw_limit','vertical_limit','forward_limit','search_yaw'}
        for row,(label,key,lo,hi,step) in enumerate(controls):
            ttk.Label(settings,text=label).grid(row=row,column=0,sticky='w',pady=3)
            self.vars[key]=tk.StringVar(value=f'{values[key]*(100 if key in self.percent else 1):g}')
            ttk.Spinbox(settings,textvariable=self.vars[key],from_=lo,to=hi,increment=step,width=7).grid(row=row,column=1,padx=(8,0))
        self.hold=tk.BooleanVar(value=bool(values['follow_and_hold']))
        ttk.Checkbutton(settings,text='Continuous Dance / Follow & Hold',variable=self.hold).grid(row=len(controls),columnspan=2,sticky='w',pady=6)
        self.edge_search_on=tk.BooleanVar(value=bool(values['edge_search_enabled']))
        ttk.Checkbutton(settings,text='Directional search: yaw 100%',variable=self.edge_search_on).grid(row=len(controls)+1,columnspan=2,sticky='w',pady=6)
        ttk.Button(settings,text='Apply + reset tracking',command=self.apply).grid(row=len(controls)+2,columnspan=2,sticky='ew')
        self.settings_note=tk.StringVar(value='Settings ready')
        ttk.Label(settings,textvariable=self.settings_note,foreground='#d9b56c',wraplength=285).grid(row=len(controls)+3,columnspan=2,sticky='w',pady=6)
        ttk.Label(settings,text='Search ends at the angle OR time limit.\n0 deg disables the turn. Heading uses TCP 9997.\nTarget loss keeps Dance ON; reacquisition is automatic.',
                  foreground='#8cbbb8',wraplength=285).grid(row=len(controls)+4,columnspan=2,sticky='w')
        ttk.Button(right,text='Reset target / spacing',command=self.reset).pack(fill='x',pady=2)
        ttk.Button(right,text='Open session logs',command=self.open_logs).pack(fill='x',pady=2)
        self.notice=tk.StringVar(value='Enter the IP shown in MSDKRemote. Start its Video Server first.')
        ttk.Label(root,textvariable=self.notice,wraplength=1250,padding=10,foreground='#d9b56c').pack(side='bottom',fill='x',before=body)
        self.settings=Settings(**self.loaded_values)
        for var in list(self.vars.values())+[self.hold,self.edge_search_on]: var.trace_add('write',self.mark_dirty)
        self.compute.trace_add('write',lambda *a:self.notice.set('Compute changes take effect on the next Connect.'))
        self.root.after(50,self.refresh)

    def read_settings(self):
        data=asdict(self.settings)
        for key,var in self.vars.items(): data[key]=float(var.get())/(100 if key in self.percent else 1)
        data['follow_and_hold']=self.hold.get()
        data['edge_search_enabled']=self.edge_search_on.get()
        return Settings(**data).validate()

    def save_config(self):
        data={'schema_version':5,'phone_ip':self.ip.get().strip(),'codec':self.codec.get(),
              'compute':self.compute.get(),'gpu_adapter':self.device.get(),'settings':asdict(self.settings),
              'axis_limits':self.current_axis_limits()}
        tmp=self.config_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(self.config_path)

    def connect(self):
        if self.session is not None or self.stopping: return
        host=self.ip.get().strip()
        try:
            # Numeric IP prevents unbounded DNS lookup in the connection worker.
            try: socket.inet_pton(socket.AF_INET,host)
            except OSError: socket.inet_pton(socket.AF_INET6,host)
            self.settings=self.read_settings(); self.save_config()
            self.session=FollowSession(self.base,host,self.codec.get(),self.settings,self.record.get(),self.compute.get(),self.device.get())
            self.session.start(); self.rates.clear(); self.mark_dirty()
            self.compute_picker.configure(state='disabled'); self.device_picker.configure(state='disabled')
            self.connect_button.configure(state='disabled'); self.disconnect_button.configure(state='normal')
            self.notice.set('Video connected. Keyboard control connects separately using Connect keyboard.')
        except Exception as exc:
            if self.session: self.session.stop(); self.session=None
            messagebox.showerror('Cannot start',str(exc))

    def disconnect(self,closing=False):
        self.stop_manual()
        if self.stopping: return
        if self.manual and self.manual.thread.is_alive():
            self.root.after(50,lambda:self.disconnect(closing)); return
        if self.session is None:
            if closing: self.root.destroy()
            return
        self.stopping=True; session=self.session
        self.disconnect_button.configure(state='disabled'); self.state.set('DISCONNECTING')
        thread=threading.Thread(target=session.stop,daemon=True); thread.start()
        def finished():
            if thread.is_alive(): self.root.after(50,finished); return
            self.session=None; self.stopping=False; self.draw_key=None
            self.connect_button.configure(state='normal')
            self.compute_picker.configure(state='readonly'); self.device_picker.configure(state='normal')
            self.state.set('DISCONNECTED'); self.intent.set('Yaw +0.000   Up +0.000\nSide +0.000  Forward +0.000')
            self.notice.set(f'Session saved: {session.output}')
            self.canvas.delete('all'); self.last_image=None; self.last_frame=None; self.last_meta=None
            self.recording_status.set('Recordings saved; open session logs')
            if closing: self.root.destroy()
        self.root.after(50,finished)

    def close(self):
        self.closing=True
        pending=getattr(self,'focus_check_id',None)
        if pending:
            try: self.root.after_cancel(pending)
            except tk.TclError: pass
            self.focus_check_id=None
        if hasattr(self,'settings'):
            try: self.save_config()
            except (OSError,ValueError,tk.TclError): pass
        self.stop_manual()
        def finish():
            if self.manual and self.manual.thread.is_alive():
                self.root.after(50,finish)
            else:
                self.disconnect(closing=True)
        finish()

    def toggle_manual(self):
        if self.manual and self.manual.thread.is_alive():
            self.stop_manual(); return
        host=self.session.receiver.host if self.session else self.ip.get().strip()
        try:
            try: socket.inet_pton(socket.AF_INET,host)
            except OSError: socket.inet_pton(socket.AF_INET6,host)
        except OSError:
            self.notice.set('Enter a numeric Phone IP before connecting keyboard.'); return
        self.manual=ManualControl(host,on_event=self.log_control,decision_provider=self.dance_decision)
        self.manual.set_axis_limits(self.current_axis_limits())
        if self.dance_selected: self.manual.start_dance()
        self.manual.start()
        self.canvas.focus_set()

    def log_control(self,event,data):
        session=self.session
        if session: session.record_control(event,data)

    def toggle_dance(self):
        self.dance_rejection=''
        self.dance_selected=not self.dance_selected
        self.dance_terminal_seen=False
        self.pressed.clear()
        if self.dance_selected:
            if self.session: self.session.reset()
            if self.manual: self.manual.start_dance()
            self.notice.set('Dance selected. Keyboard overrides while held; release keys to resume. Q / Esc releases control.')
        else:
            if self.manual: self.manual.manual_mode()
            self.notice.set('Dance OFF. Manual control remains available.')
        self.log_control('dance_selection',{'selected':self.dance_selected})
        self.root.bell()
        self.canvas.focus_set()
        self.show_control_indicator()

    def finish_terminal_dance(self,decision):
        """Only the explicitly selected one-shot mode has a terminal spacing state."""
        if decision and decision.get('continuous_dance',True):return False
        if (not decision or not self.dance_selected or not self.manual
                or self.manual.mode!='DANCE' or self.dance_terminal_seen):
            return False
        if decision.get('spacing_phase') not in ('STOPPED','SEQUENCE_DONE'):
            return False
        prediction_time=decision.get('prediction_time')
        if not isinstance(prediction_time,(int,float)) or prediction_time<=self.manual.dance_started:
            return False
        self.dance_terminal_seen=True; self.dance_selected=False; self.pressed.clear()
        self.manual.manual_mode()
        reason=decision.get('spacing_reason') or decision.get('reason') or 'terminal_stop'
        self.notice.set(f'Dance ended: {reason}. Press START DANCE to begin a new deterministic run.')
        self.log_control('dance_terminal_stop',{
            'reason':reason,'spacing_phase':decision.get('spacing_phase'),
            'policy_state':decision.get('state'),'prediction_time':prediction_time})
        self.root.bell()
        return True

    def current_axis_limits(self):
        return {axis:max(0.,min(100.,float(self.axis_limit_vars[axis].get())))/100.
                for axis in AXES}

    def change_axis_speed(self,axis,value):
        try:
            percent=float(value)
            if not math.isfinite(percent): raise ValueError('not finite')
        except (TypeError,ValueError,tk.TclError):
            percent=(self.manual.axis_limits[axis]*100 if self.manual
                     else DEFAULT_AXIS_LIMITS[axis]*100)
            self.axis_limit_vars[axis].set(percent)
            if axis in getattr(self,'axis_entry_vars',{}): self.axis_entry_vars[axis].set(f'{percent:g}')
            self.axis_limit_labels[axis].set(f'{percent:.1f}%')
            self.notice.set(f'Invalid {axis} speed. Previous value restored.')
            return
        percent=max(0.,min(100.,percent))
        self.axis_limit_vars[axis].set(percent)
        if axis in getattr(self,'axis_entry_vars',{}): self.axis_entry_vars[axis].set(f'{percent:g}')
        self.axis_limit_labels[axis].set(f'{percent:.1f}%')
        limits=self.current_axis_limits()
        if self.manual: self.manual.set_axis_limits(limits)
        self.log_control('axis_speed_changed',{'axis':axis,'percent':percent,
                                                'axis_limits':limits})

    def reject_dance(self,reason):
        self.dance_rejection=reason
        self.notice.set(reason)
        self.log_control('dance_rejected',{'reason':reason})
        self.show_control_indicator()
        self.root.bell()

    def show_control_indicator(self):
        title,detail,color=control_indicator(self.manual,self.dance_rejection,self.dance_selected)
        if (self.dance_selected and self.manual and self.manual.wanted and self.manual.enabled
                and (not self.session or self.manual.host!=self.session.receiver.host)):
            title,detail,color='DANCE ON - WAITING FOR VIDEO','Connect video to the same phone as control','#76550d'
        self.control_banner.configure(text=title,bg=color)
        self.dance_status.set(detail)
        self.dance_button.configure(text='STOP DANCE' if self.dance_selected else 'START DANCE')
        indicator=(title,detail)
        if indicator!=self.last_control_indicator:
            self.log_control('control_indicator',{'title':title,'detail':detail})
            self.last_control_indicator=indicator
        self.canvas.delete('control_badge')
        label=self.canvas.create_text(16,16,anchor='nw',text=title,fill='white',
                                      font=('Segoe UI',15,'bold'),tags='control_badge')
        bounds=self.canvas.bbox(label)
        if bounds:
            x1,y1,x2,y2=bounds
            background=self.canvas.create_rectangle(x1-7,y1-5,x2+7,y2+5,fill=color,
                                                     outline='',tags='control_badge')
            self.canvas.tag_lower(background,label)

    def enable_manual(self):
        self.dance_rejection=''
        if self.manual and self.manual.thread.is_alive() and not self.manual.stop_event.is_set():
            self.pressed.clear(); self.manual.update(set()); self.manual.enable()
            self.canvas.focus_set()

    def release_manual(self):
        self.pressed.clear()
        if self.manual: self.manual.release()

    def stop_manual(self):
        self.release_manual()
        if self.manual: self.manual.stop()

    def manual_action(self,action):
        if not self.manual or not self.manual.request(action):
            self.notice.set('Enable keyboard control (E) before takeoff or landing.')
        self.canvas.focus_set()

    def key_press(self,event):
        key=normalized_event_key(event)
        if key in ('q','escape'):
            self.release_manual(); return 'break'
        if event.widget.winfo_class() not in ('Canvas','Tk'):
            return
        pending=self.key_releases.pop(key,None)
        if pending: self.root.after_cancel(pending)
        if key in self.pressed: return 'break'
        self.pressed.add(key)
        if key in MOVEMENT_KEYS and self.manual:
            self.manual.update(self.pressed & MOVEMENT_KEYS)
        if key=='e':
            self.enable_manual(); self.pressed.add(key)
        elif key=='f': self.manual_action('takeoff')
        elif key=='r': self.manual_action('land')
        elif key=='x': self.close()
        if key in MOVEMENT_KEYS or key in ('e','f','r','x'): return 'break'

    def key_release(self,event):
        key=normalized_event_key(event)
        def released():
            self.key_releases.pop(key,None); self.pressed.discard(key)
            if self.manual: self.manual.update(self.pressed & MOVEMENT_KEYS)
        pending=self.key_releases.pop(key,None)
        if pending: self.root.after_cancel(pending)
        self.key_releases[key]=self.root.after_idle(released)

    def focus_out(self,event):
        # Tk buttons take focus on mouse-down, BEFORE their command on mouse-up.
        # Keep authority across the flight toolbar so clicking Dance can run.
        def check():
            self.focus_check_id=None
            if getattr(self,'closing',False): return
            focused=safe_focus_widget(self.root)
            if focused is self.canvas: return
            toolbar=getattr(self,'control_bar',None)
            speedbar=getattr(self,'speed_bar',None)
            viewbar=getattr(self,'view_bar',None)
            def inside(widget,ancestor):
                while widget is not None:
                    if widget is ancestor: return True
                    widget=getattr(widget,'master',None)
                return False
            if focused is not None and (inside(focused,toolbar) or inside(focused,speedbar) or inside(focused,viewbar)):
                self.pressed.clear()
                if self.manual: self.manual.update(set())
                return
            if self.manual and self.manual.wanted:
                self.log_control('control_focus_release',{'reason':'window_or_settings_focus'})
            self.release_manual()
        if getattr(self,'closing',False): return
        pending=getattr(self,'focus_check_id',None)
        if pending:
            try: self.root.after_cancel(pending)
            except tk.TclError: pass
        try: self.focus_check_id=self.root.after_idle(check)
        except tk.TclError: pass

    def apply(self):
        try:
            self.settings=self.read_settings(); self.save_config()
            if self.session: self.session.configure(self.settings)
            self.mark_dirty()
            self.notice.set('Settings saved. Tracking and spacing reset; waiting for a NEW measurement.')
        except Exception as exc: messagebox.showerror('Settings',str(exc))

    def reset(self):
        if self.session: self.session.reset(); self.notice.set('Tracker reset. A new target can be acquired.')

    def open_logs(self):
        path=self.session.output if self.session else self.base/'logs'; path.mkdir(exist_ok=True)
        if os.name=='nt': os.startfile(str(path))
        elif sys.platform=='darwin': subprocess.Popen(['open',str(path)])
        else: self.notice.set(str(path))

    def snapshot(self):
        if self.last_image is None or self.last_frame is None: return
        path=self.session.output if self.session else self.base/'logs'; path.mkdir(exist_ok=True)
        stamp=time.time_ns(); out=path/f'snapshot_{stamp}.png'
        success,encoded=cv2.imencode('.png',self.last_image)
        if success:
            out.write_bytes(encoded.tobytes())
            meta=self.last_meta or {}
            out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            self.notice.set(f'Snapshot saved: {out.name}')

    def dance_decision(self):
        s=self.session
        if (s and self.manual and not self.stopping and self.manual.host==s.receiver.host
                and not s.reset_event.is_set() and not s.model_error and s.receiver.state=='STREAMING'):
            return s.decision.get()
        return None

    def refresh(self):
        if self.manual:
            decision=self.dance_decision()
            if self.finish_terminal_dance(decision): decision=None
            self.manual.update(self.pressed & MOVEMENT_KEYS,decision)
            self.manual_status.set(self.manual.status)
            self.manual_button.configure(text='Disconnect keyboard' if self.manual.thread.is_alive() else 'Connect keyboard')
        s=self.session
        if s and not self.stopping:
            now=time.monotonic(); frame=s.receiver.frames.get(); analysis=s.analysis.get(); decision=s.decision.get()
            self.rates.append((now,s.receiver.count,s.inference_count,s.receiver.processed_count))
            elapsed=max(.01,now-self.rates[0][0]); source_fps=(s.receiver.count-self.rates[0][1])/elapsed
            video_fps=(s.receiver.processed_count-self.rates[0][3])/elapsed
            inference_fps=(s.inference_count-self.rates[0][2])/elapsed
            self.state.set(s.receiver.state if not decision else decision['state'])
            if s.receiver.error: self.state.set('VIDEO ERROR'); self.notice.set(s.receiver.error+' Disconnect, then retry.')
            elif s.model_error: self.state.set('MODEL ERROR'); self.notice.set(s.model_error)
            elif s.log.error: self.notice.set('Log write failed: '+s.log.error)
            if decision:
                age=decision.get('measurement_age_ms'); age_text='--' if age is None else f'{age:.0f} ms'
                search=decision.get('edge_search') or {}
                search_text=(f"Search: {search.get('angle_progress_degrees',0):.0f} / "
                             f"{self.settings.search_yaw_degrees:g} deg | {self.settings.edge_search_seconds:g} s\n"
                             f"{search.get('reason','--')}\n{s.heading.status}")
                result=s.result.get(); inference='--' if result is None else f"{result['inference_ms']:.1f} ms"
                compute=s.model_info.get('compute',{}) if s.model_info else {}
                compute_label=compute.get('label','loading')
                if compute.get('fallback_reason'): compute_label+=' (GPU fallback)'
                self.metrics.set(f"Model: {compute_label}\n"
                    f"YOLO: {inference} | target age: {age_text}\n"
                    f"Track source: {decision.get('track_support','--')}\n"
                    +('Short forward continuation\n' if decision.get('prediction_forward_allowed') else '')+
                    f"Spacing: {decision.get('spacing_phase','--')}\n"
                    f"Width: {100*decision.get('raw_width_ratio',0):.1f}%\n"
                    f"{decision.get('reason','')}\n{decision.get('spacing_reason','')}"
                    +'\n'+search_text
                    +('\nMovement preview stopped: RESET required' if decision.get('spacing_phase')=='STOPPED' else '')
                    +('\n'+compute['fallback_reason'][:200] if compute.get('fallback_reason') else ''))
                c=decision['intent']; self.intent.set(f"Yaw {c['yaw']:+.3f}   Up {c['vertical']:+.3f}\nSide {c['roll']:+.3f}  Forward {c['forward']:+.3f}")
            # Cap display drawing separately from inference; avoid redundant conversions.
            fid=(analysis['result']['frame'].frame_id if analysis and self.view.get()=='Analyzed frame' else frame.frame_id if frame else 0)
            displayed=(analysis['result']['frame'] if analysis and self.view.get()=='Analyzed frame' else frame)
            stale=displayed is None or now-displayed.decoded_at>self.settings.stale_seconds
            key=(fid,self.view.get(),int(now*10),self.canvas.winfo_width(),self.canvas.winfo_height())
            if key!=self.draw_key:
                image,chosen,meta=render_bundle(frame,analysis,decision,self.view.get(),self.settings.stale_seconds)
                if image is not None:
                    self.last_image=image; self.last_frame=chosen; self.last_meta=meta
                    rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    rgb.thumbnail((max(2,self.canvas.winfo_width()),max(2,self.canvas.winfo_height())),Image.Resampling.BILINEAR)
                    self.photo=ImageTk.PhotoImage(rgb)
                    self.canvas.delete('all'); self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,image=self.photo,anchor='center')
                self.draw_key=key
            decode_age='--' if frame is None else f'{(now-frame.decoded_at)*1000:.0f} ms'
            size='--' if frame is None else f'{frame.image.shape[1]}x{frame.image.shape[0]}'
            self.video_stats.set(f'{size} | Input {source_fps:.1f} FPS | Processed video {video_fps:.1f} FPS | YOLO {inference_fps:.1f} FPS | Decode age {decode_age} | Unanalyzed selected frames {s.inference_skips}')
        self.show_control_indicator()
        recorder=s.recorder if s else None
        recording=recorder.status() if recorder else None
        if recording:
            self.recording_status.set(f"{recording['state']} | {recording['mode']} | {recording['elapsed_seconds']:.1f}s | "
                                      f"frames {recording['written']} | skipped {recording['dropped']}"
                                      +(f" | {recording['error']}" if recording['error'] else ''))
        busy=bool(recorder and recorder.thread.is_alive())
        self.capture_button.configure(text='Save' if self.capture_mode.get()=='Snapshot' else 'Stop' if busy else 'Start')
        self.root.after(33,self.refresh)

    def mark_dirty(self,*args):
        try: dirty=asdict(self.read_settings())!=asdict(self.settings)
        except Exception: dirty=True
        self.settings_note.set('Changes pending - press Apply' if dirty else 'Settings applied / saved')

    def capture_action(self):
        if self.capture_mode.get()=='Snapshot': self.snapshot(); return
        try:
            if not self.session: raise RuntimeError('Connect to live video first.')
            recorder=self.session.recorder
            if recorder and recorder.thread.is_alive(): self.session.stop_recording()
            else: self.session.start_recording(self.capture_mode.get())
        except Exception as exc: messagebox.showerror('Recording',str(exc))
