"""Short continuation of a verified body-frame command, never new target proof.

No positions, speed in m/s, or obstacle clearance are inferred from a box.
Every permission expires from the original real measurement, including at send.
"""
import math
from follow_neo.transport.telemetry import fresh_heading

RECOVERY_SECONDS = .55
MAX_HEADING_CHANGE = 35.


def measured_exit(anchor,visual,now):
    """Edge departure can end Nano while a short real-source search continues.

    This never restores Nano authority, uses its new box for steering, or
    survives another veto/reset. All appearance and geometry checks must pass.
    """
    from follow_neo.control.nano_guard import finite
    e=(visual or {}).get('exit_context') or {};checks=e.get('checks') or {}
    box=e.get('box');size=e.get('frame_size');side=anchor.get('search_side',0)
    if (visual.get('reason')!='target_leaving_frame' or not side
            or e.get('prior_epoch')!=anchor.get('epoch') or visual.get('epoch')!=anchor.get('epoch',-2)+1
            or e.get('identity')!=anchor.get('identity') or e.get('generation')!=anchor.get('generation')
            or not finite(e.get('time')) or not anchor['source_time']<=e['time']<=now
            or e['time']-anchor['source_time']>.45):return False
    if (not isinstance(box,(tuple,list)) or len(box)!=4 or not all(finite(v) for v in box)
            or not isinstance(size,(tuple,list)) or len(size)!=2 or not all(finite(v) and v>0 for v in size)
            or checks.get('geometry_suspect',True) or checks.get('control_suspect',True)
            or checks.get('texture_loss_streak',1)!=0 or checks.get('appearance_unsupported',True)
            or not finite(checks.get('appearance_correlation')) or checks['appearance_correlation']<.30
            or not 0<=checks.get('visible_fraction',1)<.85):return False
    return bool(box[2]>box[0] and box[3]>box[1] and
                (box[0]<0 and (box[0]+box[2])*.5<size[0]*.10 if side<0 else
                 box[2]>size[0] and (box[0]+box[2])*.5>size[0]*.90))


class RecoveryVector:
    def __init__(self):self.anchor=None

    def remember(self,d,visual,now,enabled):
        from follow_neo.control.nano_guard import current_visual_support,sustained_real_neo,visual_agreement,finite
        identity=(d.get('target_identity') or {}).get('logical_target_id')
        if (not enabled or d.get('stale') or d.get('spacing_phase')!='APPROACH'
                or d.get('spacing_close_guard_armed') or d.get('class_support') is not None
                or self.anchor and (self.anchor['identity']!=identity or self.anchor['generation']!=d.get('generation'))):
            self.anchor=None;return
        if not d.get('accepted'):return
        # A genuinely newer stop must revoke older movement permission.
        intent=d.get('intent') or {}
        if not finite(intent.get('forward')) or intent['forward']<=0 or d.get('bbox_clipped'):
            # Alignment/edge clipping may stop the current measured command.
            # Retain the older permission without renewing its source clock;
            # only a separately checked edge departure can use it in search.
            if (self.anchor and finite(intent.get('forward'))
                    and (d.get('forward_control') or {}).get('reason') in ('Aligning with target','Target box at image edge')
                    and d.get('last_strong_measurement_time')==d.get('measurement_time')):return
            self.anchor=None;return
        source=d.get('measurement_time')
        if not finite(source) or not 0<=now-source<=.25:
            return
        if self.anchor and source<=self.anchor['source_time']:return
        if (d.get('state')!='TRACK' or not sustained_real_neo(d)
                or d.get('last_strong_measurement_time')!=source
                or not fresh_heading(d.get('heading'),now)
                or not visual_agreement(d,visual or {}).get('allowed')
                or not current_visual_support(visual,now,identity,d.get('generation'),d.get('track_box'))):return
        basis=d['intent_basis'];lateral=(d.get('centering_control') or {}).get('lateral_control') or {}
        if (not all(finite(basis.get(axis)) and basis[axis]>0 for axis in ('forward','roll'))
                or not finite(intent.get('roll',0))):
            self.anchor=None;return
        roll=intent.get('roll',0)/basis['roll'] if lateral.get('reason')=='confirmed_lateral_motion' else 0.
        options=d.get('motion_settings') or {}
        width=(d.get('frame_size') or [0])[0]
        boxes=[s['box'] for s in d['real_neo_evidence']['samples'][-3:]]
        centers=[(b[0]+b[2])*.5/width for b in boxes] if width else []
        side=(1 if all(c>=.60 for c in centers) else -1 if all(c<=.40 for c in centers) else 0) if len(centers)==3 else 0
        self.anchor=dict(source_time=source,measurement_id=d['measurement_id'],identity=identity,
            generation=d['generation'],epoch=visual['epoch'],heading_degrees=d['heading']['yaw_deg'],
            forward=min(.35,intent['forward']/basis['forward']),roll=max(-.20,min(.20,roll)),
            evidence=d['real_neo_evidence'],box=d['selected_measurement_box'],search_side=side,
            search_seconds=options.get('search_translation_seconds',RECOVERY_SECONDS),
            search_forward_fraction=options.get('search_forward_fraction',.35),
            search_side_fraction=options.get('search_side_fraction',.20))

    def snapshot(self):return dict(self.anchor) if self.anchor else None


def recovery_vector(d,visual,now,started=None):
    """Rechecked at controller, final command and while waiting for its ACK."""
    from follow_neo.control.nano_guard import finite,guard_reason,sustained_real_neo
    a=d.get('recovery_anchor') or {}
    no=lambda reason:dict(allowed=False,reason=reason,roll=0.,forward=0.)
    source=a.get('source_time');identity=(d.get('target_identity') or {}).get('logical_target_id')
    if not a or not finite(source):return no('no_verified_vector')
    searching=d.get('state')=='DIRECTIONAL_SEARCH'
    options=d.get('motion_settings') or {}
    configured_search=searching and 'search_translation_seconds' in options
    horizon=RECOVERY_SECONDS
    if searching:
        horizons=(a.get('search_seconds',RECOVERY_SECONDS),options.get('search_translation_seconds',RECOVERY_SECONDS))
        if not all(finite(v) and RECOVERY_SECONDS<=v<=1.20 for v in horizons):return no('invalid_vector_horizon')
        horizon=min(horizons)
    age=now-source
    if (not 0<=age<horizon or started is not None and source<=started):return no('vector_expired')
    departed=bool(configured_search and visual and measured_exit(a,visual,now))
    if (not visual or visual.get('mode')!='guarded' or not departed and a.get('epoch')!=visual.get('epoch')
            or a.get('identity')!=identity or identity is None or visual.get('identity')!=identity
            or a.get('generation')!=d.get('generation') or visual.get('generation')!=d.get('generation')):return no('vector_identity_changed')
    if (d.get('stale') or not d.get('confirmed') or d.get('class_support') is not None
            or d.get('selected_class_id')!=0 or d.get('selected_class_name')!='neo'
            or d.get('spacing_phase')!='APPROACH' or d.get('spacing_close_guard_armed')
            or d.get('candidate_pending') or d.get('bbox_clipped') and not departed
            or d.get('state') not in ('TRACK','COAST','DIRECTIONAL_SEARCH')):return no('vector_state')
    mtime=d.get('measurement_time')
    if d.get('accepted') and finite(mtime) and 0<=now-mtime<=.25:return no('real_measurement_available')
    if not sustained_real_neo(dict(real_neo_evidence=a.get('evidence'),target_identity=d.get('target_identity'),
            measurement_time=source,measurement_id=a.get('measurement_id'))):return no('invalid_vector_evidence')
    reason=guard_reason(visual,now)
    if not departed and reason not in (None,'semantic_expired','visual_expired','visual_deadline'):return no('visual_veto:'+str(reason))
    checks=(visual['exit_context']['checks'] if departed else visual.get('checks')) or {}
    if (checks.get('geometry_suspect',True) or checks.get('texture_loss_streak',0)>0
            or checks.get('control_suspect',True)):return no('visual_quality')
    video=d.get('video_time');heading=d.get('heading')
    if not finite(video) or not 0<=now-video<=.12 or not fresh_heading(heading,now):return no('vector_sensor_age')
    search=d.get('edge_search') or {}
    if d.get('state')=='DIRECTIONAL_SEARCH' and (not search.get('active') or search.get('candidate_guidance')
            or search.get('direction_memory')
            or search.get('verifying') or (search.get('sectors') or {}).get('phase') in ('settle','look','done')):return no('search_observation')
    f=a.get('forward');r=a.get('roll');yaw=a.get('heading_degrees')
    if not all(finite(v)for v in (f,r,yaw)) or not 0<f<=.35 or abs(r)>.20:return no('invalid_vector')
    delta=(heading['yaw_deg']-yaw+180)%360-180
    if abs(delta)>MAX_HEADING_CHANGE:return no('vector_heading_change')
    angle=math.radians(delta);decay=max(0.,min(1.,(horizon-age)/.30))
    # Rotate in the final RC command scale: GUI forward and roll limits can
    # differ. Rotating before those limits would change the requested direction.
    # This is command-space compensation, not a measured velocity estimate.
    caps=d.get('recovery_axis_limits') or dict(forward=1.,roll=1.)
    fc,rc=caps.get('forward'),caps.get('roll')
    if not all(finite(v) and 0<=v<=1 for v in (fc,rc)):return no('invalid_axis_limits')
    forward=max(0.,min(f,(f*fc*math.cos(angle)+r*rc*math.sin(angle))/fc))*decay if fc else 0.
    roll=max(-.20,min(.20,(r*rc*math.cos(angle)-f*fc*math.sin(angle))/rc))*decay if rc else 0.
    route='verified_vector_decay'
    if configured_search:
        # Brief approach to a measured bearing, never a Nano-derived trajectory.
        side=a.get('search_side',0)
        if side not in (-1,0,1):return no('invalid_search_side')
        turn=search.get('scan_yaw',0.) if search.get('phase')=='SCAN' else search.get('direction',0.)
        if not finite(turn) or not turn:return no('search_not_turning')
        forward_caps=(a.get('search_forward_fraction',0),options.get('search_forward_fraction',0))
        side_caps=(a.get('search_side_fraction',0),options.get('search_side_fraction',0))
        if (not all(finite(x) and 0<=x<=.35 for x in forward_caps)
                or not all(finite(x) and 0<=x<=.20 for x in side_caps)):
            return no('invalid_search_caps')
        forward_cap=min(forward_caps);side_cap=min(side_caps)
        if side and side*turn<0:return no('search_direction_changed')
        forward=min(f,forward_cap)*decay if fc else 0.
        roll=side*side_cap*decay if rc else 0.
        route='measured_bearing_search'
    return dict(allowed=True,reason=route,source_time=source,age_ms=age*1000,
        until=source+horizon,heading_delta_degrees=delta,decay=decay,search_side=a.get('search_side',0),
        measured_exit=departed,
        roll=roll,forward=forward,axis_limits=dict(forward=fc,roll=rc))
