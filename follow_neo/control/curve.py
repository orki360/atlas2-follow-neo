"""Early image-bearing turns and a bounded, visually verified continuation.

These are maneuvers within TRACK/COAST, not a new source of target authority.
No metric distance or target trajectory is inferred from the camera image.
"""
from collections import deque
import math

from follow_neo.transport.telemetry import fresh_heading

LEAD_SECONDS = .12
MAX_LEAD_ERROR = .10  # fraction of half the image width
MAX_CURVE_HEADING_CHANGE = 20.


class EarlyCurve:
    def __init__(self):
        self.samples=deque(maxlen=16);self.identity=None;self.active=False

    def update(self, source, error, now, identity, measured, allowed):
        no=lambda reason:dict(active=False,reason=reason,lead=0.,direction=0)
        if (not measured or identity is None or not all(isinstance(x,(float,int)) and math.isfinite(x)
                for x in (source,error,now)) or not 0<=now-source<=.25):
            self.samples.clear();self.active=False;return no('no_fresh_measurement')
        if identity!=self.identity:
            self.samples.clear();self.identity=identity;self.active=False
        if self.samples and (source<self.samples[-1][0] or source-self.samples[-1][0]>.20):
            self.samples.clear();self.active=False
        if not self.samples or source>self.samples[-1][0]:self.samples.append((source,error))
        if not allowed or len(self.samples)<3:
            self.active=False;return no('waiting_coherent_motion')
        # Three adjacent Nano frames span less than the existing 100 ms proof
        # window. Sample that SAME time horizon at either camera or YOLO rate.
        c=self.samples[-1]
        middle=[s for s in self.samples if c[0]-s[0]>=.05-1e-9]
        b=middle[-1] if middle else None
        earlier=[s for s in self.samples if b is not None and b[0]-s[0]>=.05-1e-9]
        if not earlier:
            self.active=False;return no('waiting_coherent_motion')
        a=earlier[-1]
        spans=(b[0]-a[0],c[0]-b[0]);steps=(b[1]-a[1],c[1]-b[1])
        direction=1 if error>0 else -1
        coherent=(all(.005<=dt<=.20 for dt in spans) and c[0]-a[0]>=.10-1e-9
                  and all(.003<step*direction<.15 for step in steps)
                  and (c[1]-self.samples[-2][1])*direction>0)
        speed=(c[1]-a[1])/(c[0]-a[0]) if c[0]>a[0] else 0.
        # Inward motion/reversal immediately removes the lead; repeated ticks
        # cannot add evidence. Hysteresis applies only to entry/exit thresholds.
        threshold=.08 if self.active else .12
        active=bool(coherent and abs(speed)>=threshold and abs(error)>=.025)
        self.active=active
        if not active:return no('inward_reversal_or_jitter')
        lead=direction*min(MAX_LEAD_ERROR,abs(speed)*LEAD_SECONDS)
        return dict(active=True,reason='verified_outward_motion',lead=lead,direction=direction,
                    source_time=source,measured_speed=speed,projected_error=error+lead)


class CurveContinuation:
    def __init__(self):self.anchor=None

    def remember(self,d,visual,now,enabled):
        from .nano_guard import finite,current_visual_support,sustained_real_neo,visual_agreement
        identity=(d.get('target_identity') or {}).get('logical_target_id')
        if (not enabled or not (d.get('motion_settings') or {}).get('coordinated_turns')
                or d.get('stale') or d.get('class_support') is not None
                or d.get('spacing_phase')!='APPROACH' or d.get('spacing_close_guard_armed')
                or d.get('bbox_clipped') or self.anchor and
                (identity!=self.anchor['identity'] or d.get('generation')!=self.anchor['generation'])):
            self.anchor=None;return
        if not d.get('accepted'):return
        curve=(d.get('centering_control') or {}).get('early_curve') or {}
        intent=d.get('intent') or {};basis=d.get('intent_basis') or {};source=d.get('measurement_time')
        if (d.get('state')!='TRACK' or not curve.get('active') or not d.get('confirmed')
                or not finite(source) or not 0<=now-source<=.25
                or d.get('last_strong_measurement_time')!=source or not sustained_real_neo(d)
                or not fresh_heading(d.get('heading'),now)
                or not visual_agreement(d,visual or {},now)['allowed']
                or not current_visual_support(visual,now,identity,d.get('generation'),d.get('track_box'))
                or not all(finite(intent.get(k)) and finite(basis.get(k)) and basis[k]>0 for k in ('yaw','forward'))
                or intent['forward']<=0 or intent['yaw']*curve.get('direction',0)<=0):
            self.anchor=None;return
        if self.anchor and source<=self.anchor['source_time']:return
        self.anchor=dict(source_time=source,measurement_id=d['measurement_id'],identity=identity,
            generation=d['generation'],epoch=visual['epoch'],heading_degrees=d['heading']['yaw_deg'],
            direction=curve['direction'],yaw=min(.25,abs(intent['yaw']/basis['yaw']))*curve['direction'],
            forward=min(.35,intent['forward']/basis['forward']),evidence=d['real_neo_evidence'],
            box=d['selected_measurement_box'])

    def snapshot(self):return dict(self.anchor) if self.anchor else None


def verified_curve(d,visual,now,started=None):
    """Recheck original expiry and visual/heading evidence also at TCP send."""
    from .nano_guard import finite,guard_reason,sustained_real_neo,FORWARD_BRIDGE_SECONDS,MEASUREMENT_SECONDS
    no=lambda reason:dict(allowed=False,reason=reason,yaw=0.,forward=0.)
    a=d.get('curve_anchor') or {};options=d.get('motion_settings') or {}
    source=a.get('source_time');identity=(d.get('target_identity') or {}).get('logical_target_id')
    if not a or not finite(source):return no('no_curve_anchor')
    if not 0<=now-source<FORWARD_BRIDGE_SECONDS or started is not None and source<=started:return no('curve_expired')
    if (not options.get('coordinated_turns') or d.get('stale') or not d.get('confirmed')
            or d.get('state') not in ('TRACK','COAST') or d.get('class_support') is not None
            or d.get('candidate_pending') or d.get('bbox_clipped')
            or d.get('selected_class_id')!=0 or d.get('selected_class_name')!='neo'
            or d.get('spacing_phase')!='APPROACH' or d.get('spacing_close_guard_armed')):return no('curve_state')
    if d.get('measurement_time')!=source or d.get('last_strong_measurement_time')!=source:return no('curve_measurement_changed')
    if d.get('accepted') and now-source<=MEASUREMENT_SECONDS:return no('fresh_measurement_available')
    raw_time=d.get('raw_detection_time')
    if (finite(raw_time) and raw_time>source and (d.get('detection_summary') or {}).get('target_eligible',0)>0):
        return no('unmatched_neo_candidate')
    if (not visual or visual.get('mode')!='guarded' or guard_reason(visual,now) is not None
            or identity is None or identity!=a.get('identity') or visual.get('identity')!=identity
            or a.get('generation')!=d.get('generation') or visual.get('generation')!=d.get('generation')
            or visual.get('epoch')!=a.get('epoch')
            or not 0<=visual['semantic_time']-source<=.20):return no('curve_visual_contract')
    if not sustained_real_neo(dict(real_neo_evidence=a.get('evidence'),target_identity=d.get('target_identity'),
            measurement_time=source,measurement_id=a.get('measurement_id'))):return no('curve_evidence')
    checks=visual.get('checks') or {}
    if (checks.get('geometry_suspect',True) or checks.get('control_suspect',True)
            or checks.get('texture_loss_streak',0)>0 or checks.get('visible_fraction',0)<.98
            or not finite(checks.get('appearance_correlation')) or checks['appearance_correlation']<.30):return no('curve_visual_quality')
    heading=d.get('heading');video=d.get('video_time')
    if not fresh_heading(heading,now) or not finite(video) or not 0<=now-video<=.12:return no('curve_sensor_age')
    if not finite(a.get('heading_degrees')) or abs((heading['yaw_deg']-a['heading_degrees']+180)%360-180)>MAX_CURVE_HEADING_CHANGE:
        return no('curve_heading_change')
    size=d.get('frame_size');box=visual['box']
    if not isinstance(size,(list,tuple)) or len(size)!=2 or not all(finite(v) and v>0 for v in size):return no('curve_geometry')
    w,h=size;ex=(box[0]+box[2])/w-1;ey=(box[1]+box[3])/h-1
    wx=options.get('forward_window_width',.35);wy=options.get('forward_window_height',.35)
    if not all(finite(v) and v>0 for v in (wx,wy)):return no('curve_window')
    wx=min(.65,wx);wy=min(.35,wy)
    direction=a.get('direction');yaw=a.get('yaw');forward=a.get('forward')
    if (direction not in (-1,1) or not finite(yaw) or not finite(forward)
            or not 0<yaw*direction<=.25 or not 0<forward<=.35):return no('curve_command')
    if (ex*direction<=.035 or abs(ex)>=wx or abs(ey)>=wy
            or box[0]<=.02*w or box[2]>=.98*w or box[1]<=.02*h or box[3]>=.98*h):return no('curve_alignment')
    decay=max(0.,min(1.,(FORWARD_BRIDGE_SECONDS-(now-source))/.15))
    alignment=min(1.,(wx-abs(ex))/(wx*.2),(wy-abs(ey))/(wy*.2))
    return dict(allowed=True,reason='verified_curve_decay',source_time=source,decay=decay,
                yaw=direction*min(abs(yaw),abs(ex)*.7)*decay,forward=forward*decay*alignment)
