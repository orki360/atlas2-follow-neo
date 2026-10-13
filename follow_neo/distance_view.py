"""Tk workflow for one global Mini 4 Pro distance calibration."""
from copy import deepcopy
import time
import tkinter as tk
from tkinter import ttk,messagebox
from .distance_calibration import (CAPTURE_SECONDS,SampleCapture,measured_mini,
                                   fit_global_profile,load_global_profile,estimate_distance)


class DistancePanel:
    def __init__(self,owner,config):
        self.owner=owner;self.capture=None;self.runtime=None;self.context=None
        self.points=[];self.profile=None
        load_error=None
        saved=config.get('distance_calibration')
        try:self.profile=load_global_profile(saved)
        except (ValueError,TypeError,KeyError) as exc:load_error=str(exc)
        # A malformed or incompatible old profile must not be silently erased
        # by an unrelated GUI settings save.
        owner.distance_calibration=saved if load_error else self.profile
        if self.profile:
            self.points=deepcopy(self.profile['points']);self.context=dict(self.profile['context'])
        self.tab=ttk.Frame(owner.tabs);owner.tabs.add(self.tab,text='Distance')
        canvas=tk.Canvas(self.tab,highlightthickness=0,bg='#17232d')
        scroll=ttk.Scrollbar(self.tab,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        content=ttk.Frame(canvas,padding=8);item=canvas.create_window(0,0,window=content,anchor='nw')
        content.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(item,width=e.width))
        ttk.Label(content,text='Global distance calibration / כיול גלובלי',font=('Segoe UI',12,'bold')).pack(anchor='w')
        ttk.Label(content,text='מצלמת Mini 4 Pro העוקב אל Mini 4 Pro האחר. כל המדידות מכל הכיוונים נכנסות לטבלה ולכיול אחד, ללא בחירת צד.',wraplength=305).pack(anchor='w',pady=5)
        ttk.Label(content,text='בכל מרחק אסוף כמה כיוונים, כולל זוויות ביניים. שמור על מרחק וזווית קבועים בכל איסוף של 10 שניות; סובב את הרחפן בין האיסופים. חזור בלפחות 3 מרחקים שונים.',wraplength=305).pack(anchor='w',pady=4)
        ttk.Label(content,text='Follower camera: DJI Mini 4 Pro | Zoom: 1x (fixed)',wraplength=305).pack(anchor='w',pady=4)
        ttk.Label(content,text='השאר ROI מופעל בזמן הכיול. גם זיהוי NEO שתוקן ל־Mini 4 Pro באמצעות ROI נכלל במדידה, לפי רוחב התיבה בתמונה המלאה.',wraplength=305).pack(anchor='w',pady=4)
        self.measurement_status=tk.StringVar(value='Waiting for live Mini 4 Pro detections')
        ttk.Label(content,textvariable=self.measurement_status,wraplength=305).pack(anchor='w',pady=3)
        row=ttk.Frame(content);row.pack(fill='x',pady=4)
        ttk.Label(row,text='Known distance (m)').pack(side='left')
        self.distance=tk.StringVar(value='')
        ttk.Entry(row,textvariable=self.distance,width=9).pack(side='right')
        self.collect_button=ttk.Button(content,text='Collect calibration point (10 s)',command=lambda:self.start('calibration'))
        self.collect_button.pack(fill='x')
        self.check_button=ttk.Button(content,text='Check known distance (10 s)',command=lambda:self.start('validation'))
        self.check_button.pack(fill='x',pady=3)
        ttk.Button(content,text='Cancel capture',command=self.cancel).pack(fill='x')
        self.progress=ttk.Progressbar(content,maximum=CAPTURE_SECONDS);self.progress.pack(fill='x',pady=5)
        self.status=tk.StringVar(value='Global calibration loaded. All directions use the same profile.' if self.profile else 'Collect multiple directions at each of at least 3 distances, then Save.')
        if load_error:self.status.set('Saved calibration could not be loaded: '+load_error)
        ttk.Label(content,textvariable=self.status,wraplength=305).pack(anchor='w',pady=3)
        self.table=ttk.Treeview(content,columns=('m','px','n','spread'),show='headings',height=5,selectmode='browse')
        for key,label,width in [('m','Meters',63),('px','BBOX px',78),('n','Frames',60),('spread','Spread',65)]:
            self.table.heading(key,text=label);self.table.column(key,width=width,stretch=True)
        self.table.pack(fill='x',pady=5)
        row=ttk.Frame(content);row.pack(fill='x')
        ttk.Button(row,text='Remove selected',command=self.remove).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='Clear draft',command=self.clear).pack(side='left',expand=True,fill='x')
        ttk.Button(content,text='Save global calibration / שמירת כיול גלובלי',command=self.save).pack(fill='x',pady=5)
        self.summary=tk.StringVar()
        ttk.Label(content,textvariable=self.summary,wraplength=305).pack(anchor='w',pady=4)
        ttk.Label(content,textvariable=owner.distance_text,wraplength=305,foreground='#6ad5cb').pack(anchor='w',pady=5)
        ttk.Label(content,text='המרחק משוער: רוחב הרחפן משתנה עם הזווית. הטווח המוצג מתאר את הפיזור במדידות הכיול, ואינו הבטחת דיוק. בדוק מרחק נוסף מכמה כיוונים. הכיול משמש לתצוגה בלבד.',wraplength=305).pack(anchor='w',pady=5)
        self.redraw()

    def redraw(self):
        self.table.delete(*self.table.get_children())
        for index,p in enumerate(self.points):
            self.table.insert('', 'end',iid=str(index),values=(f"{p['distance_m']:g}",f"{p['width_px']:.1f}",p['samples'],f"{p['relative_spread']:.1%}"))
        stations=len({round(p['distance_m'],3) for p in self.points})
        text=f"Global draft: {len(self.points)} captures at {stations} different distances."
        if self.profile is None:text+='\nNo saved global calibration.'
        else:
            p=self.profile
            text+=(f"\nSaved global calibration: {len(p['points'])} captures"
                   f"\nRange: {p['min_distance_m']:g}-{p['max_distance_m']:g} m"
                   f" | max fit error: {p['fit_max_error']:.1%}")
            if p['fit_max_error']>.25:
                text+='\nWide variation: use the displayed spread and check more distances.'
            if p['validations']:
                v=p['validations'][-1]
                text+=f"\nLast check: known {v['distance_m']:g} m / estimated {v['estimated_m']:.2f} m / error {v['relative_error']:.1%}"
            else:text+='\nNo independent distance check yet.'
        self.summary.set(text)

    def snapshot(self,now):
        session=self.owner.session
        if session is None or getattr(self.owner,'stopping',False):return None,None,'Connect to live video first'
        runtime=(id(session),session.generation)
        if self.runtime!=runtime:
            self.runtime=runtime
            if self.capture:self.cancel('Source / settings changed; capture cancelled')
        result=session.result.get()
        if not result or result.get('generation')!=session.generation:return None,None,'Waiting for current detector results'
        measurement,reason=measured_mini(result,now)
        frame=result['frame']
        context=dict(resolution=[frame.image.shape[1],frame.image.shape[0]],
                     model=session.model_manifest.get('sha256') or session.model_manifest['name'],
                     source=session.receiver.host)
        return measurement,context,reason

    def start(self,kind):
        try:
            if self.capture:raise ValueError('Finish or cancel the current capture first')
            now=time.monotonic();measurement,context,reason=self.snapshot(now)
            if measurement is None:raise ValueError(reason)
            if kind=='validation':
                if self.profile is None:raise ValueError('Save the global calibration before checking a distance')
                if context!=self.profile['context']:raise ValueError('Source, model or resolution differs from saved calibration')
            elif self.points and context!=self.context:
                raise ValueError('Camera source or settings changed; clear the draft to start a new calibration')
            self.capture=SampleCapture(float(self.distance.get()),now,context,kind)
            self.collect_button.configure(state='disabled');self.check_button.configure(state='disabled')
            self.status.set('Capturing new measured frames for 10 seconds...')
        except (ValueError,TypeError) as exc:messagebox.showerror('Distance calibration',str(exc),parent=self.owner.root)

    def cancel(self,reason='Capture cancelled'):
        self.capture=None;self.progress['value']=0
        self.collect_button.configure(state='normal');self.check_button.configure(state='normal')
        self.status.set(reason)

    def persist(self,profile):
        old=self.owner.distance_calibration;self.owner.distance_calibration=profile
        try:self.owner.save_config()
        except Exception:self.owner.distance_calibration=old;raise
        self.profile=profile

    def log(self,event,data):
        if self.owner.session:self.owner.session.log.write(event,data)

    def finish(self):
        capture=self.capture
        try:
            point=capture.finish()
            if capture.kind=='validation':
                updated=deepcopy(self.profile)
                updated['validations'].append(dict(distance_m=point['distance_m'],width_px=point['width_px']))
                updated=load_global_profile(updated);self.persist(updated)
                v=updated['validations'][-1]
                self.log('distance_calibration_check',dict(v,mode='global'))
                note=f"Check saved: {v['estimated_m']:.2f} m vs known {v['distance_m']:g} m; error {v['relative_error']:.1%}"
            else:
                self.context=capture.context;self.points.append(point)
                self.log('distance_calibration_point',dict(point=point,context=self.context,mode='global'))
                note=f"Global point added: {point['distance_m']:g} m, {point['samples']} frames. Turn to another angle or change distance."
            self.cancel(note);self.redraw()
        except (ValueError,OSError) as exc:self.cancel(str(exc))

    def save(self):
        try:
            if self.capture:raise ValueError('Finish or cancel capture before saving')
            profile=fit_global_profile(self.points,self.context)
            self.persist(profile);self.log('distance_calibration_saved',profile)
            self.status.set('Global calibration saved. Check another distance from several directions.')
            self.redraw()
        except (ValueError,OSError) as exc:messagebox.showerror('Distance calibration',str(exc),parent=self.owner.root)

    def remove(self):
        if self.capture:return
        selected=self.table.selection()
        if selected:
            self.points.pop(int(selected[0]));self.redraw();self.status.set('Global draft changed; Save applies the new fit.')

    def clear(self):
        if self.capture:return
        self.points=[];self.context=None;self.redraw()
        self.status.set('Global draft cleared. The saved profile remains active until Save.')

    def update(self,now):
        measurement,context,reason=self.snapshot(now)
        self.measurement_status.set(reason)
        if measurement is None:self.owner.distance_text.set('Other Mini 4 Pro distance: -- | '+reason)
        else:
            distance,quality=estimate_distance(self.profile,measurement,context)
            self.owner.distance_text.set('Other Mini 4 Pro distance: -- | '+quality if distance is None else f'Other Mini 4 Pro distance: ~{distance:.2f} m | {quality}')
        if self.capture:
            if self.owner.session is None or getattr(self.owner,'stopping',False):self.cancel('Disconnected; capture cancelled');return
            if context is not None and context!=self.capture.context:self.cancel('Source changed; capture cancelled');return
            try:self.capture.add(measurement)
            except ValueError as exc:self.cancel(str(exc));return
            elapsed=now-self.capture.started;self.progress['value']=min(CAPTURE_SECONDS,elapsed)
            self.status.set(f"{elapsed:.1f} / {CAPTURE_SECONDS:g} s | {len(self.capture.samples)} valid frames | {reason}")
            if elapsed>=CAPTURE_SECONDS:self.finish()
