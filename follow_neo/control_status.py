"""Visible control authority, distinct from perception's TRACK state."""


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
    if getattr(control,'speed',1.)==0:
        return ('DANCE ON - SPEED 0' if dance else 'MANUAL - SPEED 0'), 'Movement held at zero by speed slider', '#76550d'
    if getattr(control,'motion_hold',False)==True:
        return ('DANCE ON - MOVEMENT PAUSED' if dance else 'MANUAL - MOVEMENT PAUSED'), 'Flight action | Enable (E) to resume movement', '#76550d'
    if dance:
        detail = control.dance_status
        if detail.startswith('Keyboard override'):
            return 'DANCE ON - KEYBOARD OVERRIDE', detail, '#234f86'
        if detail.startswith('Tracking /'):
            return 'NEO DANCE ON - TRACKING', detail + ' | Q / Esc: release', '#116846'
        if detail.startswith('Sequence stopped'):
            return 'NEO DANCE ON - STOPPED', detail, '#922c36'
        return 'NEO DANCE ON - WAITING', detail + ' | Movement held at zero', '#76550d'
    return 'MANUAL CONTROL - DANCE OFF', 'Start NEO Dance to follow | Q / Esc: release', '#234f86'
