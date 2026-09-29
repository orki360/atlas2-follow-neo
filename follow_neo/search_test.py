"""An explicit, yaw-only test using the production recovery planner and gate."""
import uuid
from dataclasses import asdict
from .recovery_search import RecoverySearch,scan_command
from .telemetry import fresh_heading


class SearchTest:
    TAIL_SECONDS=2.

    def __init__(self,settings,heading,now,scenario,on_event):
        self.search=RecoverySearch();self.settings=settings;self.on_event=on_event
        self.search.start_test(now,settings,heading,scenario)
        self.run_id=uuid.uuid4().hex[:12];self.started=now;self.stopped=None
        self.reason=None;self.finished=False;self.decision=None
        self.initial_heading=heading['yaw_deg'];self.last_heading=heading
        self.offset=0.;self.travel=0.;self.minimum=0.;self.maximum=0.
        self.measurement_complete=True;self.last_sample=None;self.last_rate=None
        self.on_event('search_test_started',dict(run_id=self.run_id,scenario=scenario,
            settings=asdict(settings),initial_heading_degrees=self.initial_heading,
            scan_min_degrees=self.search.low,scan_max_degrees=self.search.high,
            deadline=self.search.until,source=heading.get('source'),
            timestamp_origin='local_request_monotonic',tail_seconds=self.TAIL_SECONDS))

    def stop(self,now,reason):
        if self.stopped is None:
            self.stopped=now;self.reason=reason;self.search.end(reason)
            self.on_event('search_test_stopped',dict(run_id=self.run_id,reason=reason,
                measured_offset_degrees=self.offset,stop_requested_at=now))

    def finish(self,now):
        if self.finished:return
        self.finished=True
        self.on_event('search_test_summary',dict(run_id=self.run_id,reason=self.reason,
            elapsed_seconds=now-self.started,initial_heading_degrees=self.initial_heading,
            measured_offset_degrees=self.offset,travel_degrees=self.travel,
            minimum_offset_degrees=self.minimum,maximum_offset_degrees=self.maximum,
            overshoot_degrees=max(0.,self.search.low-self.minimum,self.maximum-self.search.high),
            coverage_complete=len(self.search.boundaries)==2,
            measurement_complete=self.measurement_complete,
            tail_complete=self.stopped is not None and now-self.stopped>=self.TAIL_SECONDS))

    def update(self,heading,now,can_move=True):
        if self.finished:return (0.,0.,0.,0.)
        valid=fresh_heading(heading,now)
        if now-self.last_heading['time']>.5:self.measurement_complete=False
        if valid and heading['time']>self.last_heading['time']:
            dt=heading['time']-self.last_heading['time']
            delta=(heading['yaw_deg']-self.last_heading['yaw_deg']+180)%360-180
            if dt>.5 or abs(delta)>90 or heading.get('source')!=self.last_heading.get('source'):
                self.measurement_complete=False;self.last_rate=None
            else:
                self.offset+=delta;self.travel+=abs(delta);self.last_rate=delta/dt
                self.minimum=min(self.minimum,self.offset);self.maximum=max(self.maximum,self.offset)
            self.last_heading=dict(heading)
        if not can_move:self.stop(now,'control_or_operator_stop')
        search=self.search.update(now,True,False,'APPROACH',False,self.settings,heading)
        if self.stopped is None:
            if search['consumed']:self.stop(now,search['reason'])
            elif search['coverage_complete']:self.stop(now,'scan_complete')
        state='DIRECTIONAL_SEARCH' if search['active'] and self.stopped is None else 'SEARCH_PAUSED'
        self.decision=dict(prediction_time=now,state=state,accepted=False,stale=False,
            spacing_phase='SEARCH_TEST',edge_search=search,heading=heading)
        yaw=scan_command(self.decision,now,self.started) if self.stopped is None else 0.
        self.last_sample=dict(run_id=self.run_id,phase=search['phase'],reason=self.reason or search['reason'],
            heading_degrees=heading['yaw_deg'] if valid else None,
            heading_time=heading['time'] if valid else None,
            heading_age_ms=(now-heading['time'])*1000 if valid else None,
            measured_offset_degrees=self.offset,travel_degrees=self.travel,
            yaw_rate_deg_s=self.last_rate if valid else None,
            target_offset_degrees=search['scan_target_degrees'],
            scan_min_degrees=self.search.low,scan_max_degrees=self.search.high,
            requested_yaw=yaw,measurement_complete=self.measurement_complete,
            elapsed_seconds=now-self.started,tail=self.stopped is not None)
        self.on_event('search_test_sample',self.last_sample)
        if self.stopped is not None and now-self.stopped>=self.TAIL_SECONDS:self.finish(now)
        return (yaw,0.,0.,0.)
