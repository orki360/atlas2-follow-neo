"""Synchronized full-resolution rendering; no GUI state or side effects."""
import copy
import time
import cv2


def draw_box(image,box,color,label):
    h,w=image.shape[:2]
    x1,y1,x2,y2=[int(round(box[k])) for k in ('x1','y1','x2','y2')]
    x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
    cv2.putText(image,label,(x1,max(22,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)


def box_at_frame(frame,decision,now):
    if not decision or not decision.get('confirmed') or decision.get('stale'): return None
    source=decision.get('measurement_time')
    if source is None or not 0<=frame.decoded_at-source<=.300+1e-9 or now-source>.300+1e-9: return None
    if list(decision.get('frame_size',[]))!=[frame.image.shape[1],frame.image.shape[0]]: return None
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
        label='KALMAN / FRAME TIME' if confidence is None else f'KALMAN {confidence:.2f} / FRAME TIME'
        if box: draw_box(image,box,(0,210,255),label)
        metadata['display_box']=box
        title=f'LIVE frame {chosen.frame_id} | estimated box at this frame time'
    else: return None,None,None
    metadata.update(frame_id=chosen.frame_id,decoded_at=chosen.decoded_at,
                    rendered_at=now,decision=copy.deepcopy(bound))
    h,w=image.shape[:2];cv2.drawMarker(image,(w//2,h//2),(210,210,210),cv2.MARKER_CROSS,20,1)
    cv2.rectangle(image,(0,0),(w,38),(24,31,40),-1)
    cv2.putText(image,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1)
    if bound:
        status=f"{bound.get('state','--')} | spacing {bound.get('spacing_phase','--')} | PREVIEW ONLY"
        cv2.putText(image,status,(12,h-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1)
    if now-chosen.decoded_at>stale_seconds:
        cv2.rectangle(image,(0,max(0,h-48)),(w,h),(25,25,150),-1)
        cv2.putText(image,'STALE VIDEO / RESULT',(12,h-17),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    return image,chosen,metadata


def render(frame,analysis,decision,mode,stale_seconds=.75):
    image,chosen,_=render_bundle(frame,analysis,decision,mode,stale_seconds)
    return image,chosen
