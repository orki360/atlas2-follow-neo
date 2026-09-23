"""A single, measured yaw arc per target loss. Predictions cannot re-arm it."""
from collections import deque
import math
from .telemetry import fresh_heading,HEADING_MAX_AGE


class EdgeYawSearch:
    def __init__(self):self.reset()

    def reset(self):
        self.history=deque(maxlen=12);self.direction=0;self.hits=0
        self.source_time=None;self.measurement_id=None
        self.started=None;self.until=None;self.reason='insufficient_motion_history'
        self.consumed=False;self.last_heading=None;self.heading_source=None
        self.progress=0.;self.angle_limit=0.;self.heading_time=None
        self.evidence={}

    def end(self,reason):
        self.started=None;self.until=None;self.consumed=True;self.reason=reason

    def observe(self,accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings):
        if not accepted:return
        # End full-speed yaw on the first real measurement. Another arc needs
        # fresh confirmed motion history, never the previous direction alone.
        if self.started is not None or self.consumed:
            self.reset();self.reason='reacquired'
        if self.source_time is not None and source_time<=self.source_time:return
        if box is None:return
        self.source_time=source_time;self.measurement_id=measurement_id
        self.history.append((source_time,box.cx/width,box.cy/height,box.width/width))
        while self.history and source_time-self.history[0][0]>.40:self.history.popleft()
        self.direction=0;self.hits=len(self.history)
        if not confirmed or len(self.history)<3:
            self.reason='insufficient_motion_history';return
        first,last=self.history[0],self.history[-1];span=last[0]-first[0]
        if span<.08:self.reason='short_motion_history';return
        slopes=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(self.history,list(self.history)[1:]) if b[0]>a[0]]
        slopes.sort();vx=slopes[len(slopes)//2]
        travel=last[1]-first[1];side=1 if vx>0 else -1
        agree=sum(v*side>.01 for v in slopes)/len(slopes)
        edge=box.x2/width if side>0 else 1-box.x1/width
        outward=last[1]>.56 if side>0 else last[1]<.44
        exit_seconds=max(0.,(1-edge)/max(abs(vx),.001))
        self.evidence=dict(image_vx=vx,kalman_vx=k.vx/width,span_seconds=span,
                           direction_agreement=agree,estimated_exit_seconds=exit_seconds)
        if abs(vx)>=.04 and abs(travel)>=.012 and travel*side>0 and agree>=.65 and outward and exit_seconds<=1.5:
            self.direction=side;self.reason='outward_motion_history'
        else:self.reason='no_supported_horizontal_exit'

    def update(self,now,missing,stale,spacing_phase,close,settings,heading=None):
        if stale:
            if self.started is not None:self.end('video_or_result_unavailable')
            else:self.reason='video_or_result_unavailable'
        elif not settings.edge_search_enabled or settings.search_yaw_degrees<=0:
            if self.started is not None:self.end('search_disabled')
            else:self.reason='search_disabled'
        elif not missing:
            if self.started is not None:self.end('measurement_returned_or_rejected_candidate')
        elif self.started is not None:
            if now>=self.until:self.end('search_timeout')
            elif not fresh_heading(heading,now):self.end('heading_unavailable')
            elif heading.get('source')!=self.heading_source:self.end('heading_source_changed')
            elif heading['time']>self.heading_time:
                delta=(heading['yaw_deg']-self.last_heading+180)%360-180
                if abs(delta)>90:self.end('heading_discontinuity')
                else:
                    self.progress+=self.direction*delta
                    self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                    if self.progress>=self.angle_limit:self.end('search_angle_reached')
        elif not self.consumed and self.direction and self.source_time is not None:
            age=now-self.source_time
            if age>settings.edge_search_seconds:self.end('search_timeout')
            elif age>=.12:
                if not fresh_heading(heading,now):self.reason='waiting_heading'
                else:
                    self.started=now;self.until=now+settings.edge_search_seconds
                    self.angle_limit=settings.search_yaw_degrees;self.progress=0.
                    self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                    self.heading_source=heading.get('source');self.reason='directional_search'
        active=self.started is not None
        return dict(active=active,direction=self.direction if active else 0,
                    candidate_direction=self.direction,started=self.started,until=self.until,
                    source_time=self.source_time,measurement_id=self.measurement_id,
                    reason=self.reason,consumed=self.consumed,evidence=dict(self.evidence),
                    angle_degrees=settings.search_yaw_degrees,angle_progress_degrees=max(0.,self.progress),
                    heading_time=self.heading_time,heading_source=self.heading_source,
                    duration_seconds=settings.edge_search_seconds)


def yaw_override(decision,now,dance_started=None):
    search=decision.get('edge_search') or {}
    if search.get('active') is not True:return 0.
    names=('started','until','source_time','heading_time','angle_degrees','angle_progress_degrees','duration_seconds')
    if not all(isinstance(search.get(k),(int,float)) and math.isfinite(search[k]) for k in names):return 0.
    start,end,source=search['started'],search['until'],search['source_time']
    stamp=decision.get('prediction_time');direction=search.get('direction')
    if (not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
        or direction not in (-1,1) or not 0<=now-stamp<=.20
        or not source<=start<=stamp<=now<end or not 2<=search['duration_seconds']<=10
        or abs(end-start-search['duration_seconds'])>1e-6
        or not 0<=start-source<=search['duration_seconds']
        or not 0<search['angle_degrees']<=180
        or not 0<=search['angle_progress_degrees']<search['angle_degrees']
        or not 0<=now-search['heading_time']<=HEADING_MAX_AGE
        or dance_started is not None and start<dance_started
        or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH'
        or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    return float(direction)
