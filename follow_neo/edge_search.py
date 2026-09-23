"""One short yaw-only search after an observed horizontal edge exit.

An empty center-frame detection is insufficient. Direction is armed by
confirmed, outward-moving measurements at the same horizontal edge.
"""
import math


class EdgeYawSearch:
    def __init__(self): self.reset()

    def reset(self):
        self.direction=0;self.hits=0;self.source_time=None;self.measurement_id=None
        self.started=None;self.until=None;self.reason='not_armed'

    def observe(self, accepted, missing, confirmed, box, k, width, height,
                source_time, measurement_id, settings):
        if not settings.edge_search_enabled:
            self.reset();return
        if not accepted:
            if not missing: self.reset();self.reason='competing_detection'
            return
        # Any accepted real measurement ends full-speed yaw immediately.
        if self.started is not None: self.reset();self.reason='reacquired'
        side=0
        if (confirmed and box is not None and box.width/width<settings.stop_width*.60
                and box.y1>.02*height and box.y2<.98*height):
            if box.x1<.08*width and k.vx/width<-.025: side=-1
            elif box.x2>.92*width and k.vx/width>.025: side=1
        if not side:
            self.direction=0;self.hits=0;self.source_time=None
            return
        if side!=self.direction or self.source_time is None or source_time-self.source_time>.25:
            self.hits=0
        self.direction=side;self.hits+=1;self.source_time=source_time
        self.measurement_id=measurement_id;self.reason='edge_observed'

    def update(self, now, missing, stale, spacing_phase, close, settings):
        if (not settings.edge_search_enabled or stale or close
                or spacing_phase!='APPROACH'):
            self.reset();self.reason='blocked'
        elif self.started is not None:
            if now>=self.until:
                self.reset();self.reason='search_expired'
        elif (missing and self.hits>=2 and self.source_time is not None
              and 0<=now-self.source_time<=.30):
            self.started=now
            self.until=now+settings.edge_search_seconds
            self.reason='horizontal_exit'
        active=self.started is not None
        return dict(active=active,direction=self.direction if active else 0,
                    started=self.started,until=self.until,source_time=self.source_time,
                    measurement_id=self.measurement_id,reason=self.reason)


def yaw_override(decision, now, dance_started=None):
    """Independently validate the explicit full-yaw contract at send time."""
    search=decision.get('edge_search') or {}
    if search.get('active') is not True: return 0.
    start,end,stamp=(search.get('started'),search.get('until'),decision.get('prediction_time'))
    source=search.get('source_time');direction=search.get('direction')
    if (not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,end,stamp,source))
            or direction not in (-1,1) or not 0<=now-stamp<=.20
            or not source<=start<=stamp<=now<end or not 0<end-start<=2.0
            or start-source>.30 or dance_started is not None and start<dance_started
            or decision.get('accepted') or decision.get('stale')
            or decision.get('state') in ('ERROR','ABORT_HOVER')
            or decision.get('spacing_phase')!='APPROACH'
            or decision.get('spacing_close_guard_armed')):
        return 0.
    return float(direction)
