"""Independent bearing histories and bounded steering handoff; no authority.

No image work, IO, diagnostic sampling or lifetime extension lives here.
Distance/forward limits must use the unsmoothed position and yaw demand.
"""
import math

from follow_neo.shared.types import clamp
from follow_neo.control.curve import EarlyCurve

HANDOFF_SECONDS = .10
HANDOFF_YAW_RISE = 2.0  # normalized yaw units / second, only after a source switch


class BearingMotion:
    def __init__(self):
        self.key=None;self.last_measurement=None;self.error_rate=0.;self.curve=EarlyCurve()

    def update(self, source, error, now, key, valid, curve_measured, curve_allowed):
        if self.key!=key:
            self.__init__();self.key=key
        if valid and (self.last_measurement is None or source>self.last_measurement[0]):
            old=self.last_measurement;self.last_measurement=(source,error)
            if old and .005<=source-old[0]<=.20 and abs(error-old[1])<.35:
                span=source-old[0]
                derivative=clamp((error-old[1])/span,-4.,4.)
                self.error_rate+=span/(.08+span)*(derivative-self.error_rate)
            else:self.error_rate=0.
        if self.last_measurement is None or now-self.last_measurement[0]>.25:
            self.error_rate=0.
        curve=self.curve.update(source,error,now,key,curve_measured,curve_allowed)
        return self.error_rate,curve


class YawHandoff:
    """Only ease same-direction increases after a compatible estimator switch.

    Zero, reductions, reversals, identity/epoch changes and large position jumps
    bypass the easing. Rapid source switches cannot renew an active window.
    """
    def __init__(self):
        self.key=None;self.identity=None;self.time=None;self.error=None
        self.value=0.;self.until=None

    def update(self, requested, error, now, key, identity, enabled):
        if not enabled or not all(math.isfinite(x) for x in (requested,error,now)):
            self.__init__();return requested,False
        dt=0. if self.time is None else now-self.time
        compatible=(self.key is not None and self.identity==identity
                    and 0<=dt<=HANDOFF_SECONDS and abs(error-self.error)<=.15)
        changed=key!=self.key
        if (not compatible or requested*self.value<=0 or abs(requested)<=abs(self.value)
                or changed and key[0]==self.key[0]):
            self.until=None
        elif changed and (self.until is None or now>=self.until):
            self.until=now+HANDOFF_SECONDS
        out=requested
        if self.until is not None and now<self.until:
            out=math.copysign(min(abs(requested),abs(self.value)+HANDOFF_YAW_RISE*dt),requested)
        self.key=key;self.identity=identity;self.time=now;self.error=error;self.value=out
        return out,out!=requested
