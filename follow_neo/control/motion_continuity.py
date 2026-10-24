"""Revalidate an in-flight setpoint; never queue or replay movement commands."""
import math
from follow_neo.control.nano_guard import apply_visual, guard_reason, transport_visual_reason, BRIDGE_SECONDS, VISUAL_SECONDS, finite
from follow_neo.control.dance import dance_command


def compatible_setpoint(sent, allowed):
    """The actual wire value must fit the new grant, including its direction.

    One wire quantum is NOT a minimum movement boost. Values have already been
    truncated to four decimals; the tolerance only absorbs float roundoff.
    """
    return (len(sent) == len(allowed) == 4
        and all(math.isfinite(a) and math.isfinite(b)
                and (a == 0 or a*b > 0 and abs(a) <= abs(b)+1e-9)
                for a, b in zip(sent, allowed)))


def pending_motion_status(sent, latest, now, started, wire_values, limits,
                          *, require_target_policy=False, resume_after=None):
    """Keep a compatible setpoint on current proof of the same target.

    Only a newer real YOLO measurement may renew semantic evidence. Nano keeps
    the captured measurement/bridge deadline; the ACK deadline never changes.
    Identity resets, background vetoes and actual stops still cancel.
    """
    latest = apply_visual(latest, latest.get('nano'), now) if latest else None
    reason = transport_visual_reason(sent, latest, now)
    old = (sent or {}).get('nano') or {}
    new = (latest or {}).get('nano') or {}
    result = dict(reason=reason, captured_reason=reason, renewed=False,
        from_authority=(sent or {}).get('nano_authority'),
        to_authority=(latest or {}).get('nano_authority'),
        captured_visual_age_ms=(now-old['visual_time'])*1000 if finite(old.get('visual_time')) else None,
        latest_visual_age_ms=(now-new['visual_time'])*1000 if finite(new.get('visual_time')) else None)
    if reason not in ('visual_expired', 'visual_deadline', 'curve_refresh_required') or old.get('blocked', True):
        return result
    # Only a worker-generated visual deadline can be replaced. Explicit or
    # malformed shorter deadlines remain revocations.
    if reason == 'visual_deadline' and abs(old['deadline']-min(
            old['semantic_time']+BRIDGE_SECONDS, old['visual_time']+VISUAL_SECONDS)) > 1e-9:
        return result
    if not latest or latest.get('nano_authority') not in ('verified_yaw', 'yolo_only', 'bridge_yaw', 'bridge_forward', 'curve_bridge'):
        return result
    if (wire_values[3]>0 and (latest.get('spacing_close_guard_armed') or latest.get('bbox_clipped')
            or latest.get('spacing_phase')!='APPROACH')
            or (wire_values[2] or wire_values[3]) and latest.get('class_support') is not None):
        result['reason']='live_translation_revoked'
        return result
    if guard_reason(new, now) and latest.get('nano_authority') != 'yolo_only':
        return result
    stamp = (sent or {}).get('prediction_time')
    semantic = old.get('semantic_time')
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (stamp, semantic))
            or not 0 <= now-stamp <= .20 or not 0 <= now-semantic <= BRIDGE_SECONDS):
        return result
    captured_at=sent.get('nano_intent_time',stamp)
    if not finite(captured_at) or guard_reason(old,captured_at):return result
    source=latest.get('measurement_time');previous=sent.get('measurement_time')
    real_handoff=bool(latest.get('nano_authority') in ('verified_yaw','yolo_only')
        and latest.get('accepted') and latest.get('confirmed')
        and latest.get('selected_class_id')==0 and latest.get('selected_class_name')=='neo'
        and latest.get('class_support') is None
        and latest.get('last_strong_measurement_time')==source
        and finite(latest.get('measurement_id')) and finite(sent.get('measurement_id'))
        and latest['measurement_id']>sent['measurement_id']
        and finite(source) and finite(previous) and source>previous and 0<=now-source<=.25)
    # Re-evaluate the OLD decision with live visual evidence too: this keeps
    # its real-source time, forward anchor, proximity and acquisition limits.
    captured=sent
    if (sent.get('position_applied') and latest.get('position_applied')
            and source==previous and latest.get('measurement_id')==sent.get('measurement_id')):
        # Current position may support the OLD wire value only after the same
        # identity/epoch checks above, and only if BOTH grants below allow it.
        # Keep the old real measurement, intent, decision and ACK deadlines.
        captured=dict(sent,control_position=latest['control_position'])
    refreshed = apply_visual(captured, new, now)
    grants = []
    for decision in ((latest,) if real_handoff else (refreshed, latest)):
        values, gate = dance_command(decision, now, started,
            require_target_policy=require_target_policy, resume_after=resume_after)
        grants.append(tuple(v*limit for v, limit in zip(values, limits)))
        result['latest_gate' if decision is latest else 'captured_gate'] = gate
    if all(compatible_setpoint(wire_values, grant) for grant in grants):
        result.update(reason=None, renewed=True,
            support_source='new_real_neo' if real_handoff else 'current_visual_same_contract',
            authorized_measurement_time=source if real_handoff else previous)
    else:
        result['reason'] = 'live_setpoint_incompatible'
        # Component-wise intersection, never a new direction or increased axis.
        # The transport may send ONE reduction under the original ACK deadline.
        # It must still neutralize immediately on any identity/background veto.
        reduced=[]
        for axis,old_value in enumerate(wire_values):
            permitted=[grant[axis] for grant in grants]
            magnitude=min([abs(old_value)]+[abs(v) for v in permitted])
            value=(math.copysign(magnitude,old_value)
                   if old_value and all(old_value*v>0 for v in permitted) else 0.)
            reduced.append(math.trunc(value*10000)/10000)
        if any(reduced) and all(compatible_setpoint(reduced,g) for g in grants):
            result['reduction_values']=reduced
            result['reduction_source']='new_real_neo' if real_handoff else 'current_visual_same_contract'
        result['allowed_wire_values']=[list(g) for g in grants]
    return result
