"""Visible control authority, distinct from perception's TRACK state."""


def target_status_text(decision):
    tracking=decision.get('tracking',{})
    state=tracking.get('acquisition_state')
    if decision.get('stale'):return 'Waiting for current video / detection'
    if tracking.get('association_reason')=='acquisition_ambiguous':return 'Ambiguous target candidates / waiting'
    if state=='verifying':return f"Verifying target ({tracking.get('acquisition_strong_hits',0)}/2 strong measurements)"
    if state=='reacquiring':return 'Reacquiring target'
    if state=='waiting':return 'Waiting for first strong target'
    if tracking.get('candidate_pending'):return 'Reacquiring target / verifying YOLO candidate'
    return decision.get('state','Waiting for target')


def acknowledged_command_text(control,now):
    if control is None or not control.enabled or not control.wanted:
        return 'RC: no active control authority'
    ack=getattr(control,'last_acknowledged',None)
    if not ack:return 'RC: awaiting acknowledgement'
    age=now-ack['time']
    if not 0<=age<=.35:return 'RC: no recent acknowledgement'
    values=ack['values']
    return f"RC ACK {ack['mode']} | yaw {values[0]:+.3f} / forward {values[3]:+.3f} ({age*1000:.0f} ms)"


def control_indicator(control, rejection='', selected=None):
    dance=(control is not None and control.mode=='DANCE') if selected is None else selected
    if rejection:
        return 'DANCE NOT STARTED', rejection, '#922c36'
    if control is None:
        if dance: return 'DANCE ON - CONTROL DISCONNECTED', 'Connect keyboard and Enable (E) to move', '#76550d'
        return 'CONTROL DISCONNECTED', 'Connect keyboard, then Enable (E)', '#354452'
    if not control.thread.is_alive() or control.stop_event.is_set():
        if dance: return 'DANCE ON - CONTROL OFF', control.status, '#922c36'
        return 'CONTROL OFF', control.status, '#922c36' if 'ERROR' in control.status else '#354452'
    if not control.wanted:
        if dance: return 'DANCE ON - CONTROL RELEASED', 'Selection kept | Enable (E) to resume movement', '#76550d'
        return 'CONTROL RELEASED', 'Dance is OFF | Enable (E) to control', '#354452'
    if not control.enabled:
        if dance: return 'DANCE ON - ENABLING CONTROL', 'Waiting for phone acknowledgement', '#76550d'
        return 'ENABLING CONTROL', 'Waiting for phone acknowledgement', '#76550d'
    if control.mode=='SEARCH_TEST':
        return 'SEARCH TEST - DANCE OFF', control.dance_status+' | STOP TEST or Q / Esc', '#76550d'
    limits=getattr(control,'axis_limits',{})
    if isinstance(limits,dict) and limits and all(value==0 for value in limits.values()):
        if not (dance and control.dance_status.startswith(('Search /','Scan /'))):
            return ('DANCE ON - SPEED 0' if dance else 'MANUAL - SPEED 0'), 'Movement held at zero by all axis limits', '#76550d'
    if getattr(control,'motion_hold',False)==True:
        return ('DANCE ON - MOVEMENT PAUSED' if dance else 'MANUAL - MOVEMENT PAUSED'), 'Flight action | Enable (E) to resume movement', '#76550d'
    if dance:
        detail = control.dance_status
        if detail.startswith('Keyboard override'):
            return 'DANCE ON - KEYBOARD OVERRIDE', detail, '#234f86'
        if detail.startswith('Search /'):
            return 'DANCE ON - SEARCH / YAW 100%', detail + ' | Q / Esc: release', '#76550d'
        if detail.startswith('Scan /'):
            return 'DANCE ON - LEFT / RIGHT SCAN', detail + ' | Q / Esc: release', '#76550d'
        if detail.startswith('Search paused /'):
            return 'DANCE ON - SEARCH PAUSED', detail, '#76550d'
        if detail.startswith('Tracking /'):
            return 'NEO DANCE ON - TRACKING', detail + ' | Q / Esc: release', '#116846'
        if detail.startswith('Kalman recovery /'):
            return 'DANCE ON - SHORT PREDICTION', 'No fresh YOLO | Short prediction; Dance remains active | Q / Esc: release', '#76550d'
        if detail.startswith('Hover /'):
            return 'DANCE ON - HOVER / WAIT', detail + ' | Auto reacquire enabled', '#76550d'
        if detail.startswith('Sequence stopped'):
            return 'NEO DANCE ON - STOPPED', detail, '#922c36'
        return 'NEO DANCE ON - WAITING', detail + ' | Movement held at zero', '#76550d'
    return 'MANUAL CONTROL - DANCE OFF', 'Start NEO Dance to follow | Q / Esc: release', '#234f86'
