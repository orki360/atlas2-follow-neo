"""Synchronized full-resolution rendering; no GUI state or side effects."""
import copy
import time
import cv2
import numpy as np
from .prediction import COAST_SECONDS, project_center


BLUE=(255,100,20)  # OpenCV BGR: blue, shared by preview and saved video.
NEO_COLOR=(60,220,80)
MINI_COLOR=(190,80,255)
NANO_COLOR=(0,180,255)
# Display raw measurements long enough to inspect slower models. This is not
# the movement deadline or the horizon for projecting a tracked target.
RAW_DISPLAY_SECONDS=.75
RAW_DELAYED_SECONDS=.25


def detection_color(detection):
    return {0:NEO_COLOR,1:MINI_COLOR}.get(detection.get('class_id'),BLUE)


def selected_detection(detection,selected,selected_class_id,accepted=True):
    if not accepted or not selected:return False
    if detection.get('class_id') is not None and detection['class_id']!=selected_class_id:return False
    return bool(np.allclose(detection['box'],selected,atol=.01,rtol=0.))


def detection_label(detection,selected=False):
    name=detection.get('class_name') or 'YOLO'
    suffix=' SELECTED' if selected else ' DISPLAY ONLY' if detection.get('class_id')==1 else ' CANDIDATE'
    return f"{name} {detection['confidence']:.2f}{suffix}"


def draw_box(image,box,color,label,dashed=False,scale=(1.,1.),label_offset=0):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[int(round(box[k]*scale[i%2])) for i,k in enumerate(('x1','y1','x2','y2'))]
    x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
    if dashed:
        for x in range(x1,x2,16):
            cv2.line(image,(x,y1),(min(x+8,x2),y1),color,2)
            cv2.line(image,(x,y2),(min(x+8,x2),y2),color,2)
        for y in range(y1,y2,16):
            cv2.line(image,(x1,y),(x1,min(y+8,y2)),color,2)
            cv2.line(image,(x2,y),(x2,min(y+8,y2)),color,2)
    else:
        cv2.rectangle(image,(x1,y1),(x2,y2),(245,245,245),4)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
    label_y=max(22,min(h-8,y1-7+label_offset))
    cv2.putText(image,label,(x1,label_y),cv2.FONT_HERSHEY_SIMPLEX,.6,(15,15,15),4)
    cv2.putText(image,label,(x1,label_y),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)


def box_at_frame(frame,decision,now):
    if not decision or not decision.get('confirmed') or decision.get('stale'): return None
    source=decision.get('measurement_time')
    horizon=COAST_SECONDS if decision.get('prediction_display_allowed') else .300
    if source is None or not 0<=frame.decoded_at-source<=horizon+1e-9 or now-source>horizon+1e-9: return None
    if list(decision.get('frame_size',[]))!=[frame.image.shape[1],frame.image.shape[0]]: return None
    anchor=decision.get('prediction_anchor')
    if anchor is not None:
        if frame.decoded_at<anchor['time']: return None
        cx,cy,_,_=project_center(anchor,frame.decoded_at)
        return dict(x1=cx-anchor['width']/2,y1=cy-anchor['height']/2,
                    x2=cx+anchor['width']/2,y2=cy+anchor['height']/2)
    k=decision.get('kinematics')
    if not k:return None
    dt=frame.decoded_at-decision['prediction_time']
    if abs(dt)>.300: return None
    cx=k['cx']+k['vx']*dt;cy=k['cy']+k['vy']*dt
    return dict(x1=cx-k['width']/2,y1=cy-k['height']/2,x2=cx+k['width']/2,y2=cy+k['height']/2)


def draw_forward_window(image,decision):
    """Eight short grey lines; same normalized limits as centering, no masks."""
    options=(decision or {}).get('motion_settings') or {}
    fw=options.get('forward_window_width',.80);fh=options.get('forward_window_height',.70)
    if not .20<=fw<=.85 or not .20<=fh<=.80:return None
    h,w=image.shape[:2];x1=round(w*(1-fw)/2);x2=round(w*(1+fw)/2)
    y1=round(h*(1-fh)/2);y2=round(h*(1+fh)/2);length=max(8,round(min(w,h)*.025))
    for x,y,sx,sy in ((x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)):
        cv2.line(image,(x,y),(x+sx*length,y),(160,160,160),2)
        cv2.line(image,(x,y),(x,y+sy*length),(160,160,160),2)
    return dict(width=fw,height=fh,display_box=[x1,y1,x2,y2])


def render_bundle(frame,analysis,decision,mode,stale_seconds=.75,now=None,output_size=None):
    now=time.monotonic() if now is None else now
    metadata={'display_mode':mode,'detections':[],'selected_box':None,'display_box':None}
    if mode=='Analyzed frame' and analysis:
        result=analysis['result'];chosen=result['frame'];image,scale=display_image(chosen,output_size)
        bound=analysis.get('decision');age=now-chosen.decoded_at
        # The pixels and detections belong to the same source frame even when
        # inference is slow or video stops. Retain them and mark the frame age.
        selected=analysis.get('selected_box')
        selected_id=analysis.get('selected_class_id')
        selected_values=[selected[k] for k in ('x1','y1','x2','y2')] if selected else None
        for d in result['detections']:
            box=dict(zip(('x1','y1','x2','y2'),d['box']))
            is_selected=selected_detection(d,selected_values,selected_id)
            color=detection_color(d)
            draw_box(image,box,color,detection_label(d,is_selected),scale=scale,
                     label_offset=22 if d.get('class_id')==1 else 0)
        # Show the BoT estimate at the analyzed SOURCE time, not the latest
        # extrapolated location. Wall-clock age belongs in the title only.
        tracked=box_at_frame(chosen,bound,chosen.decoded_at)
        if tracked:
            tracking=(bound or {}).get('tracking') or {}
            name='BoT-SORT' if tracking.get('backend')=='BoT-SORT' else 'TRACKER'
            draw_box(image,tracked,BLUE,name+' / analyzed frame',scale=scale,label_offset=44)
        metadata.update(detections=result['detections'],selected_box=selected,display_box=selected,
                        raw_detection_time=chosen.decoded_at,raw_detection_age_ms=max(0.,age)*1000,
                        raw_detection_frame_age_ms=0.,raw_detection_display='aligned',
                        detections_drawn=len(result['detections']))
        metadata.update(bot_box=tracked,masks_drawn=False)
        title=f'ANALYZED frame {chosen.frame_id} | YOLO on this frame | age {max(0.,age)*1000:.0f} ms'
        metadata['inference_ms']=result['inference_ms']
    elif frame:
        chosen=frame;image,scale=display_image(chosen,output_size);bound=decision
        box=box_at_frame(chosen,bound,now)
        confidence=bound.get('tracking_confidence') if bound else None
        support=bound.get('track_support','yolo') if bound else 'none'
        predicted=support=='prediction'
        age=max(0,(chosen.decoded_at-bound['measurement_time'])*1000) if bound and bound.get('measurement_time') is not None else None
        candidates=bound.get('raw_detections',[]) if bound else []
        candidate_time=bound.get('raw_detection_time') if bound else None
        candidate_age=now-candidate_time if candidate_time is not None else None
        frame_age=chosen.decoded_at-candidate_time if candidate_time is not None else None
        candidate_size=bound.get('frame_size') if bound else None
        display_budget=RAW_DISPLAY_SECONDS
        if candidate_time is None:display_state='unavailable'
        elif candidate_size and list(candidate_size)!=[chosen.image.shape[1],chosen.image.shape[0]]:display_state='resolution_mismatch'
        elif min(candidate_age,frame_age)<0:display_state='future_result'
        elif max(candidate_age,frame_age)>display_budget:display_state='expired'
        elif not candidates:display_state='empty'
        else:display_state='delayed' if max(candidate_age,frame_age)>RAW_DELAYED_SECONDS else 'visible'
        if display_state not in ('visible','delayed'):candidates=[]
        delayed=display_state=='delayed'
        metadata.update(raw_detection_time=candidate_time,
                        raw_detection_age_ms=None if candidate_age is None else candidate_age*1000,
                        raw_detection_frame_age_ms=None if frame_age is None else frame_age*1000,
                        raw_detection_display=display_state,detections_drawn=len(candidates))
        selected=bound.get('selected_measurement_box') if bound else None
        selected_id=bound.get('selected_class_id') if bound else None
        for d in candidates:
            measured=dict(zip(('x1','y1','x2','y2'),d['box']))
            matched=not delayed and selected_detection(d,selected,selected_id,bound.get('accepted',True))
            raw_label=detection_label(d,matched)
            if delayed:raw_label+=f' DELAYED {max(candidate_age,frame_age)*1000:.0f} ms'
            draw_box(image,measured,detection_color(d),raw_label,dashed=delayed,scale=scale,
                     label_offset=22 if d.get('class_id')==1 else 0)
        metadata['detections']=candidates
        target_name=bound.get('selected_class_name') if bound else None
        target_candidates=[d for d in candidates if d.get('class_id')==selected_id]
        absence='NO neo DETECTION' if target_name=='neo' else 'NO DETECTION'
        label=(f"PREDICTED {age:.0f} ms / {'YOLO UNMATCHED' if target_candidates else absence}" if predicted else
               'WEAK YOLO + KALMAN' if support=='weak_yolo' else 'YOLO + KALMAN')
        if support=='class_memory':label='CLASS MEMORY / ambiguous YOLO / centering only'
        tracking=bound.get('tracking',{}) if bound else {}
        if target_name:label=target_name+' '+label
        if tracking.get('backend')=='BoT-SORT':
            label=label.replace('KALMAN','BoT-SORT')
            if tracking.get('target_id') is not None:
                label+=f" target {tracking.get('logical_target_id',tracking['target_id'])} / BoT {tracking['target_id']}"
        if target_candidates and predicted:
            cv2.putText(image,f"CANDIDATE: {tracking.get('association_reason','unmatched')} | evidence {tracking.get('candidate_hits',0)}",
                        (12,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,255),1)
        if not predicted and confidence is not None: label+=f' {confidence:.2f}'
        if box: draw_box(image,box,BLUE,label,dashed=predicted,scale=scale,
                         label_offset=44 if target_candidates else 0)
        metadata.update(track_support=support,measurement_age_ms=age)
        metadata['display_box']=box;metadata['bot_box']=box;metadata['masks_drawn']=False
        title=f'LIVE frame {chosen.frame_id} | estimated box at this frame time'
        if delayed:title=f'LIVE frame {chosen.frame_id} | YOLO DELAYED {max(candidate_age,frame_age)*1000:.0f} ms | measured positions'
    else: return None,None,None
    visual=((analysis or {}).get('nano') if mode=='Analyzed frame' else None) or (bound or {}).get('nano') or {}
    if visual and visual.get('mode')!='off':
        vt=visual.get('visual_time');vb=visual.get('box')
        historical=mode=='Analyzed frame' and visual.get('historical',False)
        aligned=vt is not None and 0<=chosen.decoded_at-vt<=.2 and (historical or 0<=now-vt<=.2)
        # Visibility and movement authority are deliberately separate. A fresh
        # visual track without recent YOLO proof is displayed as UNVERIFIED.
        if vb and aligned and visual.get('visual_active',not visual.get('blocked')):
            age_ms=(chosen.decoded_at-vt)*1000
            status='HISTORICAL' if historical else 'UNVERIFIED' if visual.get('blocked') else visual['mode'].upper()
            draw_box(image,dict(zip(('x1','y1','x2','y2'),vb)),NANO_COLOR,
                f"NANO {status} / age {age_ms:.0f} ms",dashed=True,scale=scale,label_offset=65)
            metadata['nano_box']=vb
        cv2.putText(image,f"NANO {visual.get('mode','--').upper()} | {visual.get('reason','--')}",
                    (12,image.shape[0]-65),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,190,255),1)
        metadata['nano']=dict(visual)
    metadata.update(frame_id=chosen.frame_id,decoded_at=chosen.decoded_at,
                    rendered_at=now,decision=copy.deepcopy(bound),
                    selected_class_id=bound.get('selected_class_id') if bound else None,
                    selected_class_name=bound.get('selected_class_name') if bound else None)
    h,w=image.shape[:2];cv2.drawMarker(image,(w//2,h//2),(210,210,210),cv2.MARKER_CROSS,20,1)
    metadata['forward_window']=draw_forward_window(image,bound)
    cv2.rectangle(image,(0,0),(w,38),(24,31,40),-1)
    cv2.putText(image,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
    if bound:
        authority=(decision or {}).get('control_status') or bound.get('control_status') or {}
        ack=authority.get('acknowledged') or {}
        active=authority.get('active') and 0<=time.monotonic()-authority.get('time',0)<=.35
        # Recorded rendering may happen later; the snapshot belongs to capture,
        # so use its own time when the caller supplies a frame timestamp.
        if authority.get('time') is not None:
            active=authority.get('active') and abs(authority['time']-chosen.decoded_at)<=.35
        recent_ack=active and 0<=authority.get('time',0)-ack.get('time',0)<=.35
        if authority.get('connected') and authority.get('paused'):execution='CONTROL CONNECTED - MOVEMENT PAUSED'
        elif not active:execution='CONTROL OFF - SEARCH NOT EXECUTING'
        elif authority.get('motion_hold'):execution='CONTROL ON - MOVEMENT PAUSED'
        elif authority.get('mode')=='MANUAL' or authority.get('keyboard_override'):execution='MANUAL CONTROL - SEARCH NOT EXECUTING'
        elif not recent_ack:execution='CONTROL ON - AWAITING ACK'
        else:execution=f"{ack.get('mode','--')} RC ACK yaw {ack.get('values',[0])[0]:+.3f}"
        status=f"{bound.get('state','--')} | {execution}"
        metadata['control_status']=authority
        cv2.putText(image,status,(12,h-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
        test=authority.get('search_test')
        if test and not test.get('finished'):
            line=(f"SEARCH TEST | measured {test.get('measured_offset_degrees',0):+.1f} deg | "
                  f"target {test.get('target_offset_degrees',0):+.1f} deg | {test.get('reason','starting')}")
            cv2.putText(image,line,(12,h-44),cv2.FONT_HERSHEY_SIMPLEX,.6,(80,220,220),1)
    if now-chosen.decoded_at>stale_seconds:
        cv2.rectangle(image,(0,max(0,h-48)),(w,h),(25,25,150),-1)
        cv2.putText(image,'STALE VIDEO / RESULT',(12,h-17),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    return image,chosen,metadata


def display_image(frame,output_size):
    h,w=frame.image.shape[:2]
    if output_size is None or tuple(output_size)==(w,h):return frame.image.copy(),(1.,1.)
    ow,oh=output_size
    return cv2.resize(frame.image,(ow,oh),interpolation=cv2.INTER_AREA),(ow/w,oh/h)


def render(frame,analysis,decision,mode,stale_seconds=.75):
    image,chosen,_=render_bundle(frame,analysis,decision,mode,stale_seconds)
    return image,chosen
