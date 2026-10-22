import math
import threading
from collections import deque


class TimingWindow:
    """Bounded samples; summarize at 1 Hz instead of logging every tick."""
    def __init__(self):
        self.lock=threading.Lock();self.values={}

    def add(self,name,value):
        if not math.isfinite(value) or value<0:return
        with self.lock:self.values.setdefault(name,deque(maxlen=240)).append(value)

    def snapshot(self):
        with self.lock:
            values=self.values;self.values={}
        out={}
        for name,items in values.items():
            a=sorted(items)
            if a:out[name]=dict(count=len(a),mean=sum(a)/len(a),p95=a[math.ceil(.95*len(a))-1],maximum=a[-1])
        return out
