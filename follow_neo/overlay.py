"""Synchronized full-resolution rendering; no GUI state or side effects."""
import copy
import time
import cv2
from .prediction import COAST_SECONDS, project_center


def draw_box(image,box,color,label,dashed=False):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[int(round(box[k])) for k in ('x1','y1','x2','y2')]
    x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
    if dashed:
        for x in range(x1,x2,16):
            cv2.line(image,(x,y1),(min(x+8,x2),y1),color,2)
            cv2.line(image,(x,y2),(min(x+8,x2),y2),color,2)
        for y in range(y1,y2,16):
            cv2.line(image,(x1,y),(x1,min(y+8,y2)),color,2)
            cv2.line(image,(x2,y),(x2,min(y+8,y2)),color,2)
    else: cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
    cv2.putText(image,label,(x1,max(22,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)


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
    k=decision['kinematics'];dt=frame.decoded_at-decision['prediction_time']
    if abs(dt)>.300: return None
    cx=k['cx']+k['vx']*dt;cy=k['cy']+k['vy']*dt
    return dict(x1=cx-k['width']/2,y1=cy-k['height']/2,x2=cx+k['width']/2,y2=cy+k['height']/2)


def render_bundle(frame,analysis,decision,mode,stale_seconds=.75,now=None):
    now=time.monotonic() if now is None else now
    metadata={'display_mode':mode,'detections':[],'selected_box':None,'display_box':None}
    if mode=='Analyzed frame' and analysis:
        result=analysis['result'];chosen=result['frame'];image=chosen.image.copy()
        bound=analysis.get('decision');age=now-chosen.decoded_at
        if age<=.300:
            selected=analysis.get('selected_box')
            for d in result['detections']:
                box=dict(zip(('x1','y1','x2','y2'),d['box']))
                is_selected=selected and all(abs(box[k]-selected[k])<.01 for k in box)
                draw_box(image,box,(0,210,255) if is_selected else (80,220,90),
                         f"{'SELECTED' if is_selected else 'YOLO'} {d['confidence']:.2f}")
            metadata.update(detections=result['detections'],selected_box=selected,display_box=selected)
        title=f'ANALYZED frame {chosen.frame_id} | measured YOLO boxes'
        metadata['inference_ms']=result['inference_ms']
    elif frame:
        chosen=frame;image=chosen.image.copy();bound=decision
        box=box_at_frame(chosen,bound,now)
        confidence=bound.get('tracking_confidence') if bound else None
        support=bound.get('track_support','yolo') if bound else 'none'
        predicted=support=='prediction'
        age=max(0,(chosen.decoded_at-bound['measurement_time'])*1000) if bound and bound.get('measurement_time') is not None else None
        label=(f'PREDICTED {age:.0f} ms / NO YOLO' if predicted else
               'WEAK YOLO + KALMAN' if support=='weak_yolo' else 'YOLO + KALMAN')
        if not predicted and confidence is not None: label+=f' {confidence:.2f}'
        if box: draw_box(image,box,(230,130,230) if predicted else (0,210,255),label,dashed=predicted)
        metadata.update(track_support=support,measurement_age_ms=age)
        metadata['display_box']=box
        title=f'LIVE frame {chosen.frame_id} | estimated box at this frame time'
    else: return None,None,None
    metadata.update(frame_id=chosen.frame_id,decoded_at=chosen.decoded_at,
                    rendered_at=now,decision=copy.deepcopy(bound))
    h,w=image.shape[:2];cv2.drawMarker(image,(w//2,h//2),(210,210,210),cv2.MARKER_CROSS,20,1)
    cv2.rectangle(image,(0,0),(w,38),(24,31,40),-1)
    cv2.putText(image,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
    if bound:
        status=f"{bound.get('state','--')} | spacing {bound.get('spacing_phase','--')} | VISION STATUS - control in GUI/log"
        cv2.putText(image,status,(12,h-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
    if now-chosen.decoded_at>stale_seconds:
        cv2.rectangle(image,(0,max(0,h-48)),(w,h),(25,25,150),-1)
        cv2.putText(image,'STALE VIDEO / RESULT',(12,h-17),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    return image,chosen,metadata


def render(frame,analysis,decision,mode,stale_seconds=.75):
    image,chosen,_=render_bundle(frame,analysis,decision,mode,stale_seconds)
    return image,chosen
