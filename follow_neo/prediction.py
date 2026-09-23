"""Shared limits for short, explicitly labelled Kalman continuation."""
COAST_SECONDS = .65
FADE_START = .12
FORWARD_SECONDS = .25
COAST_YAW = .25
COAST_VERTICAL = .20
STABLE_FORWARD_SECONDS = .55


def fade(age, end=COAST_SECONDS):
    return max(0., min(1., (end-age)/(end-FADE_START)))


def continuation_threshold(confidence):
    # Keep the user's acquisition threshold. Only a confirmed, spatially
    # matched track can use this small lower-confidence band (C++ floor .25).
    return min(confidence, max(.25, confidence*.8))


def forward_fade(age):
    # Preserve a short, steady bridge before reducing translation.
    return max(0.,min(1.,(STABLE_FORWARD_SECONDS-age)/(.55-.20)))


def project_center(anchor, now):
    """Shared source-time trajectory: bounded displacement, damped velocity."""
    import math
    age=max(0.,now-anchor['time'])
    steady=anchor.get('damp_after',.10)
    tau=anchor.get('decay_tau',.16)
    tail=max(0.,age-steady)
    decay=math.exp(-tail/tau)
    elapsed=min(age,steady)+tau*(1-decay)
    result=[]
    for position,velocity,limit in (('cx','vx','limit_x'),('cy','vy','limit_y')):
        displacement=anchor[velocity]*elapsed
        bound=anchor[limit]
        result.append((anchor[position]+max(-bound,min(bound,displacement)),
                       anchor[velocity]*decay if abs(displacement)<bound else 0.))
    return tuple(float(v) for v in (result[0][0],result[1][0],result[0][1],result[1][1]))
