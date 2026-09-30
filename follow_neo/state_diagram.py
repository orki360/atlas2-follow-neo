"""Read-only live state diagram. No control commands, timers or worker threads."""
import math
import tkinter as tk
from tkinter import ttk

BG = '#101c27'
INACTIVE = '#243746'
PREVIEW = '#24547a'
CURRENT = '#14694f'
PAUSED = '#855d16'
ERROR = '#923d47'
STATES = ('WAIT_VIDEO','ERROR','WAIT_TARGET','REACQUIRE','COAST','TRACK',
          'HOVER_WAIT','DIRECTIONAL_SEARCH','SEARCH_PAUSED')
PHASES = ('IDLE','BOOST','SCAN','DONE')


def recent(stamp, now, limit=.35):
    return isinstance(stamp,(int,float)) and math.isfinite(stamp) and 0<=now-stamp<=limit


def number(value, format_spec='+.1f'):
    return format(value,format_spec) if isinstance(value,(int,float)) and math.isfinite(value) else '--'


def diagram_snapshot(decision, control, selected, connected, now, error=None):
    """Describe reported policy independently of permission to send movement."""
    decision=decision or {};control=control or {}
    mode=control.get('mode','MANUAL')
    fresh=recent(decision.get('prediction_time'),now)
    stale=bool(decision) and (not fresh or decision.get('stale',False))
    state=decision.get('state') if connected else None
    reason=decision.get('reason','Waiting for video / inference' if connected else 'Connect video to show live states')
    if error or state=='ERROR':state='ERROR';reason=error or reason
    elif connected and (not decision or stale):
        state='WAIT_VIDEO'
        reason='Video / decision unavailable or stale' if stale else reason
    if state not in STATES:state=None
    search=decision.get('edge_search') or {}
    phase=search.get('phase') if connected and fresh and not stale and state!='ERROR' else None
    test=control.get('search_test') or {}
    testing=mode=='SEARCH_TEST' and not test.get('finished',False)
    if testing:
        # A test has its own planner; never present it as a Dance search.
        state=None;phase=test.get('phase') if recent(control.get('time'),now) else None
        reason=test.get('reason') or 'Search test; Dance is off'
    if phase not in PHASES:phase=None
    control_fresh=recent(control.get('time'),now)
    active=bool(control.get('connected') and control.get('active') and control_fresh)
    paused=bool(control.get('paused') or control.get('motion_hold'))
    dance=bool(selected and mode=='DANCE')
    if not control.get('connected'):status='CONTROL DISCONNECTED'
    elif paused:status='CONNECTED | MOVEMENT PAUSED'
    elif not control_fresh:status='CONNECTED | CONTROL STATUS STALE'
    elif not active:status='CONNECTED | CONTROL NOT ENABLED'
    elif testing:status='SEARCH TEST | DANCE OFF'
    elif control.get('keyboard_override'):status='KEYBOARD OVERRIDE'
    elif dance:status='DANCE CONTROL ENABLED'
    else:status='MANUAL | DANCE OFF'
    color=ERROR if state=='ERROR' else PAUSED if paused else CURRENT if active and dance and not control.get('keyboard_override') else PREVIEW
    if state in ('WAIT_VIDEO','SEARCH_PAUSED'):color=PAUSED
    held=bool(search.get('paused') or search.get('verifying') or state in ('SEARCH_PAUSED','REACQUIRE'))
    phase_color=PAUSED if held or paused else color
    if testing:phase_color=CURRENT if active and not paused else PAUSED
    ack=control.get('acknowledged') or {}
    values=ack.get('values') or []
    ack_text=('Last RC ACK: yaw '+number(values[0],'+.3f') if values and recent(ack.get('time'),now) else 'Last RC ACK: none recent')
    if phase in ('BOOST','SCAN'):
        source=test if testing else search
        offset=source.get('measured_offset_degrees') if testing else source.get('scan_offset_degrees')
        target=source.get('target_offset_degrees') if testing else source.get('scan_target_degrees')
        if not testing and phase=='BOOST':
            target=search.get('scan_max_degrees') if search.get('direction',0)>0 else search.get('scan_min_degrees')
        angles='Yaw '+number(offset)+' deg | target '+number(target)+' deg'
        reason=reason+' / '+str(search.get('reason','')) if not testing else reason
    else:angles='Search yaw: --'
    return dict(state=state,phase=phase,color=color,phase_color=phase_color,status=status,
                reason=str(reason).replace('_',' '),angles=angles,ack=ack_text,
                spacing=decision.get('spacing_phase','--') if connected and fresh and not stale and not testing else '--',
                heading='SEARCH TEST PLANNER' if testing else 'SEARCH PLANNER'+(' / HELD' if held and phase in ('BOOST','SCAN') else ''),
                label='SEARCH TEST' if testing else state or 'NO LIVE DANCE STATE')


class LiveStateDiagram(ttk.Frame):
    """Create canvas items once; refresh from the GUI thread at at most 10 Hz."""
    def __init__(self,parent):
        super().__init__(parent)
        self.last_update=None;self.last_view=None;self.last_state=None
        self.status=tk.Label(self,text='CONTROL DISCONNECTED',bg=INACTIVE,fg='white',
                             font=('Segoe UI',10,'bold'),pady=5)
        self.columnconfigure(0,weight=1);self.rowconfigure(2,weight=1)
        self.status.grid(row=0,column=0,sticky='ew')
        self.current=ttk.Label(self,text='NO LIVE DANCE STATE',font=('Segoe UI',12,'bold'),anchor='center')
        self.current.grid(row=1,column=0,sticky='ew',pady=(5,2))
        self.canvas=tk.Canvas(self,width=390,height=385,bg=BG,highlightthickness=0,takefocus=0)
        self.canvas.grid(row=2,column=0,sticky='nsew')
        self.boxes={};self.labels={};self.base_coordinates={}
        self._draw()
        self.canvas.bind('<Configure>',self._resize)
        self.details=ttk.Label(self,wraplength=380,justify='left')
        self.details.grid(row=3,column=0,sticky='ew',pady=(5,2))
        self.transition=ttk.Label(self,text='Last transition: --',wraplength=380,foreground='#94a9ba')
        self.transition.grid(row=4,column=0,sticky='ew',pady=2)
        ttk.Label(self,text='Green: Dance enabled   Blue: preview\nAmber: held / waiting   Red: error\nHighlights show state, not proof of aircraft motion.',
                  foreground='#94a9ba',wraplength=380).grid(row=5,column=0,sticky='ew',pady=3)

    def _node(self,key,x,y,width=160,height=34,label=None):
        self.boxes[key]=self.canvas.create_rectangle(x-width/2,y-height/2,x+width/2,y+height/2,
            fill=INACTIVE,outline='#456070',width=1)
        self.labels[key]=self.canvas.create_text(x,y,text=label or key,fill='#b5c7d4',font=('Segoe UI',9,'bold'))

    def _arrow(self,*points):
        self.canvas.create_line(*points,fill='#527184',width=1.3,arrow='last',arrowshape=(6,7,3))

    def _draw(self):
        c=self.canvas
        c.create_text(195,14,text='DANCE / MAIN TRANSITIONS',fill='#7c9aaa',font=('Segoe UI',9,'bold'))
        self._arrow(95,69,95,86)
        self._arrow(175,105,215,105)
        self._arrow(295,124,295,147)
        self._arrow(215,165,175,165)
        self._arrow(95,184,95,207)
        self._arrow(155,184,230,207)
        self._arrow(215,225,175,225)
        self._arrow(375,225,384,225,384,105,375,105)
        self._arrow(95,207,7,207,7,78,295,78,295,86)
        self._arrow(270,244,238,263)
        self._arrow(220,265,250,244)
        self._node('WAIT_VIDEO',95,51)
        self._node('ERROR',295,51)
        self._node('WAIT_TARGET',95,105)
        self._node('REACQUIRE',295,105)
        self._node('COAST',95,165)
        self._node('TRACK',295,165)
        self._node('HOVER_WAIT',95,225)
        self._node('DIRECTIONAL_SEARCH',295,225)
        self._node('SEARCH_PAUSED',195,280)
        c.create_rectangle(5,308,385,379,outline='#456070',dash=(3,3))
        self.planner_title=c.create_text(195,319,text='SEARCH PLANNER',fill='#7c9aaa',font=('Segoe UI',9,'bold'))
        for x1,x2 in ((92,108),(187,203),(282,298)):self._arrow(x1,350,x2,350)
        for x,phase in zip((50,145,240,335),PHASES):self._node('phase_'+phase,x,350,width=82,height=30,label=phase)
        self._arrow(50,366,50,375,240,375,240,366)
        for item in c.find_all():self.base_coordinates[item]=c.coords(item)

    def _resize(self,event):
        scale=max(1,event.width)/390
        yscale=max(1,event.height)/385
        for item,coords in self.base_coordinates.items():
            self.canvas.coords(item,*[value*(scale if i%2==0 else yscale) for i,value in enumerate(coords)])
        font_size=8 if yscale<.8 else 9
        for item in self.labels.values():self.canvas.itemconfigure(item,font=('Segoe UI',font_size,'bold'))
        for label in (self.details,self.transition):label.configure(wraplength=max(150,event.width-10))

    def update_state(self,decision,control,selected,connected,now,error=None):
        if self.last_update is not None and 0<=now-self.last_update<.1:return
        self.last_update=now
        view=diagram_snapshot(decision,control,selected,connected,now,error)
        if view==self.last_view:return
        state_key=(view['label'],view['phase'])
        if state_key!=self.last_state:
            if self.last_state is not None:
                def name(pair):return pair[0]+(' / '+pair[1] if pair[1] else '')
                self.transition.configure(text='Last transition: '+name(self.last_state)+' -> '+name(state_key))
            self.last_state=state_key
        previous=self.last_view
        self.last_view=view
        self.status.configure(text=view['status'],bg=view['color'])
        self.current.configure(text=view['label'].replace('_',' '))
        if previous is None or any(previous[key]!=view[key] for key in ('state','phase','color','phase_color')):
            selected_keys={view['state']:view['color'],'phase_'+str(view['phase']):view['phase_color']}
            for key,box in self.boxes.items():
                active=key in selected_keys
                self.canvas.itemconfigure(box,fill=selected_keys.get(key,INACTIVE),outline='#d8f4ee' if active else '#456070',width=2 if active else 1)
                self.canvas.itemconfigure(self.labels[key],fill='white' if active else '#b5c7d4')
        self.canvas.itemconfigure(self.planner_title,text=view['heading'])
        self.details.configure(text='Reason: '+view['reason']+'\n'+view['angles']+'\nSpacing: '+str(view['spacing'])+' | '+view['ack'])
