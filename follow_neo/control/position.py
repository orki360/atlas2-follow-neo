"""Select a fast steering position without manufacturing detector evidence.

Nano owns the centre only. BoT/YOLO retain identity, dimensions, source times,
distance and acquisition. This first stage leaves loss/search budgets intact.
"""
from dataclasses import replace

from .nano_guard import (MEASUREMENT_SECONDS, VISUAL_SECONDS, finite,
                         guard_reason, visual_agreement)


class PositionSelector:
    """Immediate safe fallback, but require two distinct frames for re-entry.

    Waiting for Nano never holds movement: the current BoT position is used.
    Candidate frames still pass every original semantic/visual check below.
    """
    def __init__(self):
        self.key=None;self.nano_active=False;self.was_nano=False;self.candidate=None

    def select(self, decision, kinematics, visual, now, enabled=True):
        selected,info=select_position(decision,kinematics,visual,now,enabled)
        identity=(decision.get('target_identity') or {}).get('logical_target_id')
        key=(decision.get('generation'),identity)
        if not enabled or key!=self.key or identity is None:
            self.__init__();self.key=key
        info=dict(info,generation=key[0],identity=identity,handoff_enabled=bool(enabled and identity is not None))
        if info['source']!='nanotrack':
            self.nano_active=False;self.candidate=None
            return selected,info
        current=(info['epoch'],info['frame_id'],info['source_time'])
        if self.was_nano and not self.nano_active:
            old=self.candidate
            if (old is None or current[0]!=old[0] or current[1]<old[1] or current[2]<old[2]
                    or current[2]-old[2]>.10):
                self.candidate=current;old=current
            if current[1]==old[1] or current[2]<=old[2]:
                return kinematics,dict(source='botsort',reason='nano_reentry_confirming',
                    size_source='yolo_botsort',generation=key[0],identity=identity,
                    handoff_enabled=True,motion_candidate=info)
        self.nano_active=self.was_nano=True;self.candidate=None
        return selected,info


def select_position(decision, kinematics, visual, now, enabled=True):
    fallback = dict(source='botsort', reason='disabled', size_source='yolo_botsort')
    no = lambda reason: (kinematics, dict(fallback, reason=reason))
    if not enabled:
        return no('disabled')
    source = decision.get('measurement_time')
    identity = (decision.get('target_identity') or {}).get('logical_target_id')
    if (not decision.get('accepted') or not decision.get('confirmed')
            or decision.get('stale') or decision.get('class_support') is not None
            or decision.get('selected_class_id') != 0 or decision.get('selected_class_name') != 'neo'
            or not finite(source) or not 0 <= now-source <= MEASUREMENT_SECONDS
            or decision.get('last_strong_measurement_time') != source):
        return no('waiting_real_neo')
    if not visual or visual.get('mode') != 'guarded':
        return no('nano_not_guarded')
    reason = guard_reason(visual, now)
    if reason:
        return no(reason)
    if (identity is None or visual.get('identity') != identity
            or decision.get('generation') is None
            or visual.get('generation') != decision['generation']
            or visual.get('epoch') is None):
        return no('nano_identity_mismatch')
    if (not finite(visual.get('score')) or visual['score'] < .65
            or not finite(visual.get('frame_id'))
            or visual['visual_time'] < source or visual['semantic_time'] > source):
        return no('nano_position_unverified')
    checks = visual.get('checks') or {}
    if (checks.get('geometry_suspect', True) or checks.get('control_suspect', True)
            or checks.get('texture_loss_streak', 0) > 0 or checks.get('visible_fraction', 0) < .98
            or not finite(checks.get('appearance_correlation')) or checks['appearance_correlation'] < .30):
        return no('nano_position_quality')
    agreement = visual_agreement(decision, visual, now)
    if not agreement['allowed']:
        return no(agreement['reason'])
    box = visual['box']
    cx, cy = (box[0]+box[2])*.5, (box[1]+box[3])*.5
    # Width/height/velocity remain the detector's values. Image-bearing motion
    # is differentiated separately at actual Nano source times by centering.
    return replace(kinematics, cx=cx, cy=cy), dict(
        source='nanotrack', reason='verified_visual_position', size_source='yolo_botsort',
        source_time=visual['visual_time'], frame_id=visual['frame_id'], center=[cx, cy],
        epoch=visual['epoch'], generation=visual['generation'], identity=identity,
        measurement_time=source, measurement_id=decision.get('measurement_id'),
        visual_max_age_seconds=visual.get('control_max_age_seconds', VISUAL_SECONDS),
        agreement=agreement['reason'])


def position_guard_reason(decision, visual, now):
    """Recheck the position used to compute a command, including at TCP send."""
    position = (decision or {}).get('control_position') or {}
    if position.get('source') != 'nanotrack':
        return None
    if not visual or visual.get('mode') != 'guarded':
        return 'position_contract_changed'
    if (position.get('identity') is None
            or position['identity'] != (decision.get('target_identity') or {}).get('logical_target_id')
            or position.get('generation') != decision.get('generation')
            or any(position.get(k) != visual.get(k) for k in ('identity', 'generation', 'epoch'))):
        return 'position_identity_changed'
    if (position.get('measurement_time') != decision.get('measurement_time')
            or position.get('measurement_id') != decision.get('measurement_id')):
        return 'position_measurement_changed'
    stamp = position.get('source_time')
    limit = position.get('visual_max_age_seconds')
    if (not finite(stamp) or not finite(limit) or not .04 <= limit <= VISUAL_SECONDS
            or not finite(visual.get('visual_time')) or visual['visual_time'] < stamp):
        return 'invalid_position_time'
    if not 0 <= now-stamp <= limit:
        return 'visual_expired'
    reason = guard_reason(visual, now)
    if reason:
        return reason
    source = position.get('measurement_time')
    if not finite(source) or not 0 <= now-source <= MEASUREMENT_SECONDS:
        return 'position_measurement_expired'
    return None
