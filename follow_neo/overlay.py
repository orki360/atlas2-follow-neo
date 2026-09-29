"""Synchronized full-resolution rendering; no GUI state or side effects."""
import copy
import time
import cv2
import numpy as np
from .prediction import COAST_SECONDS, project_center


BLUE=(255,100,20)  # OpenCV BGR: blue, shared by preview and saved video.


def draw_box(image,box,color,label,dashed=False,scale=(1.,1.)):
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
    cv2.putText(image,label,(x1,max(22,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,(15,15,15),4)
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


def render_bundle(frame,analysis,decision,mode,stale_seconds=.75,now=None,output_size=None):
    now=time.monotonic() if now is None else now
    metadata={'display_mode':mode,'detections':[],'selected_box':None,'display_box':None}
    if mode=='Analyzed frame' and analysis:
        result=analysis['result'];chosen=result['frame'];image,scale=display_image(chosen,output_size)
        bound=analysis.get('decision');age=now-chosen.decoded_at
        if age<=.300:
            selected=analysis.get('selected_box')
            for d in result['detections']:
                box=dict(zip(('x1','y1','x2','y2'),d['box']))
                is_selected=selected and all(abs(box[k]-selected[k])<.01 for k in box)
                for contour in d.get('segments',[]):
                    points=np.asarray(contour,np.float32)*np.asarray(scale)
                    if len(points)>=3:cv2.polylines(image,[points.astype(np.int32)],True,BLUE,1)
                draw_box(image,box,BLUE,
                         f"{'SELECTED' if is_selected else 'YOLO'} {d['confidence']:.2f}",scale=scale)
            metadata.update(detections=result['detections'],selected_box=selected,display_box=selected)
        title=f'ANALYZED frame {chosen.frame_id} | measured YOLO boxes'
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
        fresh_candidates=(candidate_time is not None and 0<=chosen.decoded_at-candidate_time<=.25 and now-candidate_time<=.25)
        if not fresh_candidates:candidates=[]
        selected=bound.get('selected_measurement_box') if bound else None
        for d in candidates:
            measured=dict(zip(('x1','y1','x2','y2'),d['box']))
            matched=bool(selected and np.allclose(d['box'],selected,atol=.01))
            draw_box(image,measured,(60,220,80) if matched else (0,180,255),
                     f"{'MEASURED' if matched else 'CANDIDATE'} {d['confidence']:.2f}",scale=scale)
        metadata['detections']=candidates
        label=(f"PREDICTED {age:.0f} ms / {'YOLO UNMATCHED' if candidates else 'NO DETECTION'}" if predicted else
               'WEAK YOLO + KALMAN' if support=='weak_yolo' else 'YOLO + KALMAN')
        tracking=bound.get('tracking',{}) if bound else {}
        if tracking.get('backend')=='BoT-SORT':
            label=label.replace('KALMAN','BoT-SORT')
            if tracking.get('target_id') is not None:
                label+=f" target {tracking.get('logical_target_id',tracking['target_id'])} / BoT {tracking['target_id']}"
        if candidates and predicted:
            cv2.putText(image,f"CANDIDATE: {tracking.get('association_reason','unmatched')} | evidence {tracking.get('candidate_hits',0)}",
                        (12,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,255),1)
        if not predicted and confidence is not None: label+=f' {confidence:.2f}'
        if box: draw_box(image,box,BLUE,label,dashed=predicted,scale=scale)
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
        authority=(decision or {}).get('control_status') or bound.get('control_status') or {}
        ack=authority.get('acknowledged') or {}
        active=authority.get('active') and 0<=time.monotonic()-authority.get('time',0)<=.35
        # Recorded rendering may happen later; the snapshot belongs to capture,
        # so use its own time when the caller supplies a frame timestamp.
        if authority.get('time') is not None:
            active=authority.get('active') and abs(authority['time']-chosen.decoded_at)<=.35
        recent_ack=active and 0<=authority.get('time',0)-ack.get('time',0)<=.35
        if not active:execution='CONTROL OFF - SEARCH NOT EXECUTING'
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
