"""Visual support is not a semantic observation or collision avoidance.

This module has no sockets. Deadlines use source-frame monotonic time, never
the time at which a worker happens to finish. Only real YOLO evidence renews
semantic authority; a hard visual loss resets the tracker identity.
"""
import math
from collections import deque

MEASUREMENT_SECONDS = .250
BRIDGE_SECONDS = .400
VISUAL_SECONDS = .100
FORWARD_BRIDGE_SECONDS = .350
ZERO = dict(yaw=0., vertical=0., roll=0., forward=0.)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


class ForwardAnchorHistory:
    """Bounded source-time history; a newer stop invalidates older permission."""
    def __init__(self):
        self.items = deque(maxlen=8)

    def remember(self, anchor, now):
        source = anchor['time']
        if self.items and (anchor['identity'] != self.items[-1]['identity']
                           or source < self.items[-1]['time']):
            self.items.clear()
        if not anchor['allowed']:
            self.items.clear()
        if self.items and self.items[-1]['time'] == source:
            self.items.pop()
        self.items.append(dict(anchor))
        self.prune(now)

    def prune(self, now):
        while self.items and not 0 <= now-self.items[0]['time'] < FORWARD_BRIDGE_SECONDS:
            self.items.popleft()

    def snapshot(self):
        return [dict(a) for a in self.items]


def appearance_unsupported(checks):
    """Weak appearance needs coherent foreground motion, not a high Nano score.

    Missing LK points are unknown evidence. Good appearance needs no flow;
    blur can still be supported by coherent motion distinct from the background.
    This pauses authority, without deleting the visual tracker/template.
    """
    ref = checks.get('reference_std')
    corr = checks.get('appearance_correlation')
    textured = finite(ref) and ref > 8
    weak = not finite(corr) or corr < .30
    flow = (checks.get('flow_points', 0) >= 6
            and checks.get('background_points', 0) >= 8
            and checks.get('foreground_flow_spread_px', math.inf) < 2.5
            and checks.get('box_flow_error_px', math.inf) <= 8
            and checks.get('relative_motion_px', 0) > 2.)
    return bool(textured and weak and not flow)


def visual_quality_reason(checks):
    if appearance_unsupported(checks):
        return 'visual_appearance_unsupported'
    if checks.get('control_suspect'):
        return 'visual_quality_suspect'
    return None


def compatible_forward_anchor(decision, visual, now, identity):
    """Pair Nano with an already authorized source, never a future measurement.

    The newest measurement still has veto power and caps the older command.
    Keeping the older source also keeps its original expiry, even when YOLO
    has advanced and Nano has not validated that new source frame yet.
    """
    latest = decision.get('verified_forward_anchor') or {}
    if not latest.get('allowed'):
        return None, 'latest_forward_not_allowed'
    if latest.get('identity') != identity:
        return None, 'forward_identity_mismatch'
    source = latest.get('time')
    if not finite(source) or not 0 <= now-source < FORWARD_BRIDGE_SECONDS:
        return None, 'forward_source_expired'
    if not finite(latest.get('forward')) or latest['forward'] <= 0:
        return None, 'invalid_forward_anchor'
    # A newer weak/class-memory measurement cannot strengthen an old command.
    if decision.get('last_strong_measurement_time') != decision.get('measurement_time'):
        return None, 'weak_measurement'
    candidates = [latest] + list(decision.get('verified_forward_history') or [])
    candidates = [a for a in candidates if a.get('identity') == identity
                  and finite(a.get('time')) and a['time'] <= source
                  and 0 <= visual['semantic_time']-a['time'] <= .20
                  and 0 <= now-a['time'] < FORWARD_BRIDGE_SECONDS]
    if not candidates:
        return None, 'no_source_aligned_forward_anchor'
    anchor = max(candidates, key=lambda a: a['time'])
    if not anchor.get('allowed') or not finite(anchor.get('forward')) or anchor['forward'] <= 0:
        return None, 'aligned_forward_not_allowed'
    return dict(anchor, forward=min(anchor['forward'], latest['forward'])), 'source_aligned'


def overlap(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = max(0., a[2]-a[0])*max(0., a[3]-a[1])+max(0., b[2]-b[0])*max(0., b[3]-b[1])-area
    return area/union if union > 0 else 0.


class RealNeoEvidence:
    """Short source-frame history of independent, unambiguous real detections.

    Nano never writes this history. A missing/weak/competing frame breaks it;
    repeated controller ticks cannot add votes or refresh its timestamp.
    """
    def __init__(self):
        self.items=deque(maxlen=4)
        self.reason='waiting_real_neo'

    def observe(self,detections,accepted,confirmed,box,identity,source,frame_id,size,threshold):
        strong=[d for d in detections if d.class_id==0 and d.class_name=='neo'
                and finite(d.confidence) and d.confidence>=max(.75,threshold)]
        selected=None if box is None else list(vars(box).values())
        valid=(accepted and confirmed and identity is not None and finite(source) and finite(frame_id)
            and selected is not None and all(finite(x) for x in selected)
            and box.width>0 and box.height>0 and len(strong)==1
            and overlap(list(vars(strong[0].box).values()),selected)>=.9)
        if valid:
            valid=not any(d is not strong[0] and d.confidence>=.45
                          and overlap(list(vars(d.box).values()),selected)>.3 for d in detections)
        if not valid:
            self.items.clear();self.reason='ambiguous_or_missing_real_neo';return
        old=self.items[-1] if self.items else None
        if old and frame_id==old['frame_id'] and source==old['time']:return
        if old:
            a=old['box'];dt=source-old['time']
            coherent=(identity==old['identity'] and frame_id>old['frame_id'] and 0<dt<=.25
                and .5<=box.width/(a[2]-a[0])<=2 and .5<=box.height/(a[3]-a[1])<=2
                and abs(box.cx-(a[0]+a[2])*.5)<=max(size[0]*.12,2*box.width)
                and abs(box.cy-(a[1]+a[3])*.5)<=max(size[1]*.12,2*box.height))
            if not coherent:self.items.clear()
        self.items.append(dict(time=source,frame_id=frame_id,identity=identity,box=selected))
        self.reason='consistent_real_neo' if len(self.items)>=3 else 'verifying_real_neo'

    def snapshot(self):
        return dict(reason=self.reason,samples=[dict(a,box=list(a['box'])) for a in self.items])


def sustained_real_neo(decision):
    """Check the actual votes, not a cached boolean or a visual-track score."""
    items=(decision.get('real_neo_evidence') or {}).get('samples') or []
    if len(items)<3:return False
    identity=(decision.get('target_identity') or {}).get('logical_target_id')
    last=items[-1]
    if (last.get('time')!=decision.get('measurement_time')
            or last.get('frame_id')!=decision.get('measurement_id')):return False
    for a in items:
        b=a.get('box')
        if (a.get('identity')!=identity or not finite(a.get('time'))
                or not finite(a.get('frame_id')) or not isinstance(b,(list,tuple)) or len(b)!=4
                or not all(finite(v) for v in b) or b[2]<=b[0] or b[3]<=b[1]):return False
    if not .12<=last['time']-items[0]['time']<=.75:return False
    return all(0<b['time']-a['time']<=.25 and b['frame_id']>a['frame_id']
               for a,b in zip(items,items[1:]))


def strong_anchor(controller, result, settings):
    """Match original class/mask evidence to the accepted BoT identity."""
    t = controller.tracker
    if (controller.rejected or not t.confirmed or t.accepted is None
            or controller.class_support is not None or t.measurement_weak):
        return None
    box = list(vars(t.accepted.box).values())
    threshold = max(settings.confidence, settings.new_track_confidence)
    matches = [d for d in result['detections'] if d.get('class_id') == 0
               and d.get('class_name') == 'neo' and d['confidence'] >= threshold
               and overlap(d['box'], box) >= .7]
    if len(matches) != 1:
        return None
    chosen = matches[0]
    competing = [d for d in result['detections'] if d is not chosen
                 and overlap(d['box'], chosen['box']) > .5
                 and d['confidence'] >= chosen['confidence']-.08]
    if competing:
        return None
    return dict(frame=result['frame'], detection=chosen,
                identity=t.logical_target_id, generation=result['generation'])


def continuity_detection(result, evidence, identity, settings):
    """Only a new, unique real NEO detection can repair short association loss."""
    if not evidence or identity is None or evidence.get('identity') != identity:return None
    frame=result['frame']
    if (evidence.get('generation')!=result.get('generation')
            or evidence.get('source_time')!=frame.decoded_at
            or evidence.get('frame_id')!=frame.frame_id):return None
    threshold=max(.75,settings.confidence,settings.new_track_confidence)
    candidates=[d for d in result['detections'] if d.get('class_id')==0 and d.get('class_name')=='neo'
                and d.get('confidence',0)>=threshold and overlap(d['box'],evidence['box'])>=.65]
    if len(candidates)!=1:return None
    chosen=candidates[0]
    if any(d is not chosen and overlap(d['box'],chosen['box'])>.3
           and d.get('confidence',0)>=min(settings.confidence,.45) for d in result['detections']):return None
    return dict(box=list(chosen['box']),frame_id=frame.frame_id,source_time=frame.decoded_at,
                identity=identity,reason='real_yolo_with_visual_continuity')


def guard_reason(visual, now):
    """An enforced contract must fail closed, including on malformed data."""
    if not visual or visual.get('mode') != 'guarded':
        return None
    if visual.get('blocked', True):
        # A soft appearance warning must never hide a hard loss/identity veto.
        reason=visual.get('reason', 'visual_unavailable')
        if reason not in ('appearance_unverified','visual_appearance_unsupported','visual_supported',
                'waiting_strong_neo','waiting_visual_update','verifying_identity','semantic_expired',
                'visual_expired','visual_deadline','nano_error','visual_unavailable'):
            return reason
        return visual_quality_reason(visual.get('checks') or {}) or visual.get('reason', 'visual_unavailable')
    box=visual.get('box')
    if (not isinstance(box,(list,tuple)) or len(box)!=4
            or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in box)
            or box[2]<=box[0] or box[3]<=box[1]):
        return 'invalid_visual_box'
    if visual.get('identity') is None:return 'visual_identity_missing'
    for key in ('semantic_time', 'visual_time', 'deadline'):
        value = visual.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 'invalid_visual_time'
    if not 0 <= now-visual['semantic_time'] <= BRIDGE_SECONDS:
        return 'semantic_expired'
    age_limit=visual.get('control_max_age_seconds',VISUAL_SECONDS)
    if not finite(age_limit) or not .04<=age_limit<=VISUAL_SECONDS:return 'invalid_visual_age_limit'
    if not 0 <= now-visual['visual_time'] <= age_limit:
        return 'visual_expired'
    if now > visual['deadline']:
        return 'visual_deadline'
    return visual_quality_reason(visual.get('checks') or {})


def current_visual_support(visual,now,identity,generation,box):
    """Current camera evidence for delayed measured motion, never new identity."""
    if (not visual or visual.get('mode')!='guarded' or identity is None or generation is None
            or visual.get('identity')!=identity or visual.get('generation')!=generation
            or guard_reason(visual,now) is not None):return False
    checks=visual.get('checks') or {}
    if (checks.get('geometry_suspect',True) or checks.get('control_suspect',True)
            or checks.get('texture_loss_streak',0)>0 or checks.get('visible_fraction',0)<.98
            or not finite(checks.get('appearance_correlation')) or checks['appearance_correlation']<.30):return False
    if isinstance(box,dict):box=[box.get(k) for k in ('x1','y1','x2','y2')]
    return bool(isinstance(box,(list,tuple)) and len(box)==4 and all(finite(v) for v in box)
                and box[2]>box[0] and box[3]>box[1] and overlap(box,visual['box'])>=.25)


def lateral_visual_required(decision):
    lateral=((decision or {}).get('centering_control') or {}).get('lateral_control') or {}
    return bool(lateral.get('uses_visual_support'))


def visual_agreement(decision,visual,now=None):
    """Compare original frames, then bound the displacement to the live frame.

    This only permits fresh real YOLO authority through soft Nano doubt. A
    historical match alone never authenticates a newer Nano box on background.
    The short extrapolation checks agreement; it cannot renew any deadline.
    """
    box=decision.get('selected_measurement_box');vbox=visual.get('box')
    valid=lambda b:isinstance(b,(list,tuple)) and len(b)==4 and all(finite(v) for v in b) and b[2]>b[0] and b[3]>b[1]
    if not valid(box) or not valid(vbox):return dict(allowed=False,reason='invalid_box')
    direct=overlap(box,vbox)
    info=dict(allowed=direct>=.25,reason='direct_overlap',direct_iou=direct)
    if info['allowed']:return info
    aligned=visual.get('source_alignment') or {}
    identity=(decision.get('target_identity') or {}).get('logical_target_id')
    if (not sustained_real_neo(decision) or aligned.get('frame_id')!=decision.get('measurement_id')
            or aligned.get('source_time')!=decision.get('measurement_time')
            or aligned.get('identity')!=identity or aligned.get('generation')!=decision.get('generation')
            or aligned.get('epoch')!=visual.get('epoch') or not valid(aligned.get('box'))):
        return dict(info,reason='missing_source_alignment')
    source=decision['measurement_time'];stamp=visual.get('visual_time')
    now=decision.get('prediction_time') if now is None else now
    before=aligned.get('before_time');after=aligned.get('after_time')
    ids=[aligned.get('before_frame_id'),aligned.get('frame_id'),aligned.get('after_frame_id'),visual.get('frame_id')]
    if (not all(finite(v) for v in (stamp,before,after,now)) or stamp>now or not 0<=stamp-source<=.15
            or not 0<=source-before<=.05 or not 0<=after-source<=.05
            or not all(finite(v) for v in ids) or not ids[0]<=ids[1]<=ids[2]<=ids[3]):
        return dict(info,reason='alignment_time')
    same=overlap(box,aligned['box']);info.update(source_iou=same,delta_ms=(stamp-source)*1000)
    if same<.50:return dict(info,reason='source_disagreement')
    items=decision['real_neo_evidence']['samples'][-3:]
    times=[v['time']-source for v in items];mean=sum(times)/len(times)
    denom=sum((t-mean)**2 for t in times)
    if denom<=0:return dict(info,reason='motion_history')
    # Two axes, independently bounded by target size; a table below the drone
    # cannot pass merely because its horizontal position is plausible.
    residual=[]
    for axis in (0,1):
        centers=[(v['box'][axis]+v['box'][axis+2])*.5 for v in items]
        mid=sum(centers)/len(centers)
        speed=sum((t-mean)*(c-mid) for t,c in zip(times,centers))/denom
        predicted=(box[axis]+box[axis+2])*.5+speed*(stamp-source)
        residual.append(abs((vbox[axis]+vbox[axis+2])*.5-predicted))
    widths=[box[2]-box[0],box[3]-box[1]]
    ratios=[(vbox[i+2]-vbox[i])/widths[i] for i in (0,1)]
    agrees=(residual[0]<=max(8.,widths[0]*.45) and residual[1]<=max(6.,widths[1]*.60)
            and all(.65<=ratio<=1.5 for ratio in ratios))
    return dict(info,allowed=agrees,reason='source_aligned_motion' if agrees else 'live_motion_disagreement',
                center_residual_px=residual,size_ratios=ratios)


def independent_yolo_proof(decision,visual,*,sustained=False,now=None):
    """Fresh real detections may survive soft Nano doubt, never a hard conflict."""
    if (visual.get('reason') not in ('visual_supported','appearance_unverified','visual_appearance_unsupported')
            or decision.get('generation') is None or decision['generation']!=visual.get('generation')
            or visual.get('identity')!=(decision.get('target_identity') or {}).get('logical_target_id')
            or decision.get('raw_detection_time')!=decision.get('measurement_time')
            or decision.get('bbox_clipped')):return False
    checks=visual.get('checks') or {}
    if (checks.get('geometry_suspect',True)
            or checks.get('texture_loss_streak',0)>0 or checks.get('visible_fraction',0)<.98):return False
    # Pixel flow on a blurred drone may actually measure background texture.
    # It can veto Nano, but may not veto three consistent real NEO observations.
    # Geometry, texture loss, identity, class conflict and box disagreement stay
    # mandatory for both routes. This never grants Nano-only authority.
    if checks.get('background_like_motion',False) or checks.get('box_flow_error_px',0)>8:
        if not sustained or not sustained_real_neo(decision):return False
    box=decision.get('selected_measurement_box');vbox=visual.get('box')
    if not all(isinstance(b,(list,tuple)) and len(b)==4 and all(finite(v) for v in b)
               and b[2]>b[0] and b[3]>b[1] for b in (box,vbox)):return False
    if not visual_agreement(decision,visual,now)['allowed']:return False
    raw=decision.get('raw_detections') or []
    threshold=max(.75,decision.get('detector_acquisition_threshold',.75))
    candidates=[d for d in raw if d.get('class_id')==0 and d.get('class_name')=='neo'
                and finite(d.get('confidence')) and d['confidence']>=threshold]
    if len(candidates)!=1 or overlap(candidates[0]['box'],box)<.9:return False
    if any(d is not candidates[0] and d.get('confidence',0)>=.45
           and overlap(d['box'],box)>.3 for d in raw):return False
    return True


def apply_visual(decision, visual, now):
    """Leave all YOLO/BoT evidence fields untouched, even in guarded mode."""
    if visual:
        visual=dict(visual,control_max_age_seconds=(decision.get('motion_settings') or {}).get(
            'nano_visual_max_age_seconds',VISUAL_SECONDS))
    out = dict(decision, nano=visual)
    out.setdefault('base_intent',dict(decision.get('intent',ZERO)))
    out['intent']=dict(out['base_intent'])
    out.pop('nano_authority',None);out.pop('nano_block_reason',None)
    out.pop('nano_forward_source_time',None)
    out.pop('nano_handoff',None);out.pop('nano_intent_time',None)
    out.pop('nano_soft_veto_overridden',None)
    from follow_neo.control.pursuit import recovery_vector
    out['recovery_motion']=recovery_vector(out,visual,now)
    decision=out
    if (lateral_visual_required(decision) and not current_visual_support(visual,now,
            (decision.get('target_identity') or {}).get('logical_target_id'),decision.get('generation'),
            decision.get('track_box'))):
        out['intent']['roll']=0.
    if not visual or visual.get('mode') != 'guarded':
        return out
    out['nano_intent_time']=now
    mtime=decision.get('measurement_time')
    measurement_current=finite(mtime) and 0<=now-mtime<=MEASUREMENT_SECONDS
    diagnostics=dict(evaluated_at=now,
        measurement_age_ms=(now-mtime)*1000 if finite(mtime) else None,
        semantic_time=visual.get('semantic_time'),
        latest_forward_source_time=(decision.get('verified_forward_anchor') or {}).get('time'),
        forward_reason='not_bridging')
    if measurement_current:
        diagnostics['visual_agreement']=visual_agreement(decision,visual,now)
    out['nano_handoff']=diagnostics
    identity = (decision.get('target_identity') or {}).get('logical_target_id')
    reason = guard_reason(visual, now)
    if visual.get('identity') != identity or not decision.get('confirmed'):
        reason = reason or 'identity_unconfirmed'
    if decision.get('stale'):
        reason = reason or 'video_or_detector_stale'
    # Search is independently bounded by real detections, heading, arc and time.
    # Nano losing the target must not veto this independent recovery mechanism.
    search=decision.get('edge_search') or {}
    if (not decision.get('stale') and decision.get('confirmed')
            and decision.get('state')=='DIRECTIONAL_SEARCH' and search.get('active')
            and search.get('phase') in ('SCAN','BOOST')):
        out['nano_authority']='search';return out
    if out['recovery_motion']['allowed']:
        out['nano_authority']='recovery_vector';return out
    yolo_fresh=bool(decision.get('accepted') and decision.get('confirmed')
        and not decision.get('stale') and decision.get('class_support') is None
        and decision.get('selected_class_id')==0 and decision.get('selected_class_name')=='neo'
        and measurement_current
        and decision.get('last_strong_measurement_time')==mtime
        and decision.get('state') in ('TRACK','REACQUIRE'))
    unavailable={'waiting_strong_neo','waiting_visual_update','verifying_identity','semantic_expired',
                 'visual_expired','visual_deadline','nano_error','visual_unavailable'}
    if reason in unavailable and yolo_fresh:
        out['nano_authority']='yolo_only';return out
    if (reason in ('visual_appearance_unsupported','appearance_unverified') and yolo_fresh
            and independent_yolo_proof(decision,visual,sustained=True,now=now)):
        out.update(nano_authority='yolo_only',nano_soft_veto_overridden=reason)
        diagnostics['forward_reason']='fresh_yolo_over_soft_nano_doubt'
        diagnostics['yolo_evidence']='sustained' if sustained_real_neo(decision) else 'single_with_visual_agreement'
        return out
    if reason:
        diagnostics['forward_reason']=reason
        out.update(intent=dict(ZERO), nano_authority='blocked', nano_block_reason=reason,
                   prediction_forward_allowed=False, prediction_recovery=False)
        return out
    if (yolo_fresh and mtime!=visual.get('semantic_time')
            and independent_yolo_proof(decision,visual,sustained=True,now=now)):
        # YOLO and Nano finish independently. A new genuine measurement need
        # not be demoted to the loss bridge while Nano checks that same frame.
        out['nano_authority']='yolo_only'
        diagnostics.update(forward_reason='fresh_yolo_pending_nano_anchor',
                           bridge_reason='independent_real_measurement')
        return out
    # Between real measurements the latest visual position can steer yaw. It
    # cannot supply size, distance, acquisition hits or a search direction.
    box = visual.get('box')
    width = decision['frame_size'][0]
    ex = ((box[0]+box[2])*.5-width*.5)/(width*.5)
    yaw = max(-.25, min(.25, ex*.7)) if abs(ex) > .035 else 0.
    yaw *= max(0., min(1., (BRIDGE_SECONDS-(now-visual['semantic_time']))/.15))
    intent = dict(decision['intent'])
    # Even accepted=True may be retained from an earlier tick. A visual bridge
    # may only decay a forward command already authorized by real YOLO.
    bridge = (not measurement_current or not decision.get('accepted') or decision.get('class_support') is not None
              or decision.get('measurement_time') != visual.get('semantic_time')
              or decision.get('state') not in ('TRACK', 'REACQUIRE'))
    diagnostics['bridge_reason']=('measurement_expired_at_send' if not measurement_current
        else 'source_alignment' if mtime!=visual.get('semantic_time')
        else 'measurement_unavailable' if bridge else 'fresh_measurement')
    if bridge:
        # A current real measurement keeps the damped yaw controller. Switching
        # to Nano's proportional yaw on every valid frame used to discard that
        # controller's braking/lead and change gain at each source handoff.
        intent['yaw'] = yaw*decision['intent_basis']['yaw']
        intent.update(vertical=0., roll=0., forward=0.)
        anchor,forward_reason=compatible_forward_anchor(decision,visual,now,identity)
        diagnostics['forward_reason']=forward_reason
        anchor=anchor or {}
        checks=visual.get('checks') or {}
        source=anchor.get('time')
        box=visual['box'];height=decision['frame_size'][1]
        options=decision.get('motion_settings') or {}
        window_x=min(.35,options.get('forward_window_width',.35))
        window_y=min(.35,options.get('forward_window_height',.35))
        clear=bool(box[0]>.02*width and box[2]<.98*width and box[1]>.02*height and box[3]<.98*height)
        if (anchor.get('allowed') and anchor.get('identity')==identity
                and isinstance(source,(float,int)) and 0<=now-source<FORWARD_BRIDGE_SECONDS
                and 0<=visual['semantic_time']-source<=.20
                and decision.get('state') in ('TRACK','COAST','REACQUIRE')
                and decision.get('spacing_phase')=='APPROACH' and not decision.get('spacing_close_guard_armed')
                and not decision.get('bbox_clipped') and decision.get('class_support') is None
                and clear and abs(ex)<window_x and abs((box[1]+box[3])/height-1)<window_y
                and not checks.get('control_suspect',True) and not checks.get('geometry_suspect',True)
                and isinstance(anchor.get('forward'),(float,int)) and math.isfinite(anchor['forward'])
                and anchor['forward']>0
                and checks.get('visible_fraction',0)>=.98 and (checks.get('appearance_correlation') or 0)>=.30):
            # Reuse a verified forward command; Nano cannot estimate distance or
            # accelerate. This deadline stays tied to the last real measurement.
            decay=max(0.,min(1.,(FORWARD_BRIDGE_SECONDS-(now-source))/.15))
            intent['forward']=min(anchor['forward'],.35*decision['intent_basis']['forward'])*decay
            out['nano_forward_source_time']=source
            diagnostics.update(forward_reason='bounded_forward',forward_source_time=source,
                forward_source_age_ms=(now-source)*1000,
                semantic_to_forward_ms=(visual['semantic_time']-source)*1000,
                used_history=source!=diagnostics['latest_forward_source_time'])
        elif anchor:
            diagnostics['forward_reason']=('spacing_or_edge' if decision.get('spacing_phase')!='APPROACH'
                or decision.get('spacing_close_guard_armed') or decision.get('bbox_clipped') else
                'class_or_state' if decision.get('class_support') is not None
                or decision.get('state') not in ('TRACK','COAST','REACQUIRE') else
                'visual_alignment' if not clear or abs(ex)>=window_x or abs((box[1]+box[3])/height-1)>=window_y else
                'visual_quality')
    out.update(intent=intent, nano_authority=('bridge_forward' if intent['forward']>0 else 'bridge_yaw') if bridge else 'verified_yaw',
               prediction_forward_allowed=False)
    return out


def transport_visual_reason(sent, latest, now):
    """Check the captured permission AND live veto before/while awaiting ACK."""
    old = (sent or {}).get('nano')
    new = (latest or {}).get('nano')
    if (sent or {}).get('recovery_motion',{}).get('allowed'):
        from follow_neo.control.pursuit import recovery_vector
        if (not latest or not recovery_vector(sent,old,now)['allowed']
                or not recovery_vector(latest,new,now)['allowed']
                or (sent.get('recovery_anchor') or {}).get('source_time')!=(latest.get('recovery_anchor') or {}).get('source_time')):
            return 'recovery_vector_revoked'
    if not (old and old.get('mode') == 'guarded' or new and new.get('mode') == 'guarded'):
        return None
    if (lateral_visual_required(sent) and (sent.get('intent') or {}).get('roll',0.)!=0
            and (not latest or not current_visual_support(new,now,
                (sent.get('target_identity') or {}).get('logical_target_id'),sent.get('generation'),
                latest.get('track_box')))):
        return 'visual_lateral_expired'
    independent=('yolo_only','search','recovery_vector')
    if (sent or {}).get('nano_authority') in independent:
        if not latest or latest.get('stale') or latest.get('nano_authority')=='blocked':return 'visual_control_blocked'
        if (sent.get('target_identity') or {}).get('logical_target_id')!=(latest.get('target_identity') or {}).get('logical_target_id'):
            return 'visual_identity_changed'
        if not old or not new or old.get('mode')!=new.get('mode'):return 'visual_contract_changed'
        if old.get('generation')!=new.get('generation'):return 'visual_generation_changed'
        return None
    if not old or not new or old.get('mode') != new.get('mode'):
        return 'visual_contract_changed'
    if (old.get('epoch'), old.get('generation'), old.get('identity')) != (new.get('epoch'), new.get('generation'), new.get('identity')):
        return 'visual_epoch_changed'
    if (sent or {}).get('nano_authority')=='bridge_forward':
        source=sent.get('nano_forward_source_time')
        if (not isinstance(source,(float,int)) or not math.isfinite(source)
                or not 0<=now-source<FORWARD_BRIDGE_SECONDS):return 'visual_forward_expired'
        if (latest.get('spacing_phase')!='APPROACH' or latest.get('spacing_close_guard_armed')
                or latest.get('bbox_clipped') or latest.get('class_support') is not None
                or latest.get('state') not in ('TRACK','COAST','REACQUIRE')):
            return 'visual_forward_revoked'
        anchor=(latest or {}).get('verified_forward_anchor') or {}
        if anchor and (not anchor.get('allowed') or anchor.get('identity')!=old.get('identity')):
            return 'visual_forward_revoked'
        if latest.get('nano_authority')=='bridge_yaw':
            return 'visual_forward_revoked'
    return guard_reason(old, now) or guard_reason(new, now) or (
        'visual_control_blocked' if (latest or {}).get('nano_authority') == 'blocked' else None)
