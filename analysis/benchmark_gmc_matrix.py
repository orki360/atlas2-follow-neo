"""Offline, fixed-pixel GMC matrix. No GUI, aircraft, transport or live settings.

Uses a native Full-HD recording. Letterbox every requested canvas, mask padding,
scale pixel tolerances from the production 640-wide geometry. Cache YOLO once.
Common background-flow residuals are consistency proxies, NOT ground truth.
"""
import argparse
import base64
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import platform
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from follow_neo.perception.camera_motion import CameraMotion
from follow_neo.perception.botsort_motion import BoTSORTTracker
from follow_neo.perception.detector import OnnxDroneDetector
from follow_neo.perception.models import resolve_model,V6_MODEL
from follow_neo.shared.types import Box,Detection
from follow_neo.vendor.botsort.basetrack import BaseTrack

SIZES=[(1920,1080),(960,544),(640,360),(480,270),(423,544),(213,1080)]
COUNTS=[300,350,400,450]
WINDOWS=[(6.,26.),(32.,50.)]

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def stats(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),maximum=float(a.max())) if len(a) else dict(n=0)

def geometry(size):
    w,h=size;s=min(w/1920,h/1080);rw,rh=round(1920*s),round(1080*s)
    return rw,rh,(w-rw)//2,(h-rh)//2

def odd(value,minimum=3):
    n=max(minimum,round(value));return n if n%2 else n+1

class Variant(CameraMotion):
    def __init__(self,size,corners):
        super().__init__();self.size=size;self.corners=corners;self.record={}

    def apply(self,image,detections=None):
        begin=time.perf_counter();w,h=self.size;rw,rh,ox,oy=geometry(self.size)
        scale=rw/640.;sx=rw/image.shape[1];sy=rh/image.shape[0]
        small=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
        content=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        gray=np.full((h,w),114,np.uint8);gray[oy:oy+rh,ox:ox+rw]=content
        mask=np.zeros_like(gray);mask[oy:oy+rh,ox:ox+rw]=255
        # Padding has no features. Keep feature neighborhoods away from its edge.
        if ox or oy:
            border=max(2,round(3*scale))
            mask[:oy+border]=0;mask[oy+rh-border:]=0
            mask[:,:ox+border]=0;mask[:,ox+rw-border:]=0
        margin=max(1,round(3*scale))
        if detections is not None:
            for box in detections:
                x1,y1,x2,y2=np.asarray(box[:4])*[sx,sy,sx,sy]+[ox,oy,ox,oy]
                cv2.rectangle(mask,(max(0,int(x1)-margin),max(0,int(y1)-margin)),
                              (min(w-1,int(x2)+margin),min(h-1,int(y2)+margin)),0,-1)
        points=cv2.goodFeaturesToTrack(gray,maxCorners=self.corners,qualityLevel=.01,
                                      minDistance=max(1.,7*scale),mask=mask,blockSize=odd(3*scale))
        identity=np.eye(2,3);warp=identity;self.status='insufficient_features';self.inliers=0;good_count=0
        if self.previous is not None and self.points is not None and len(self.points)>=12:
            try:
                window=odd(21*scale,5)
                current,valid,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,winSize=(window,window))
                back,bvalid,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,current,None,winSize=(window,window))
                good=((valid.ravel()!=0)&(bvalid.ravel()!=0)&
                      (np.linalg.norm(back-self.points,axis=2).ravel()<1.5*scale))
                # Check only actual image content, not the letterbox.
                xy=current.reshape(-1,2)
                if ox or oy:
                    good &= (xy[:,0]>=ox)&(xy[:,0]<ox+rw)&(xy[:,1]>=oy)&(xy[:,1]<oy+rh)
                before,after=self.points[good],current[good];good_count=len(before)
                if good_count>=12:
                    estimate,inliers=cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,
                        ransacReprojThreshold=2*scale,maxIters=500)
                    self.inliers=0 if inliers is None else int(inliers.sum())
                    if estimate is not None and np.isfinite(estimate).all():
                        # Evaluate translation in content coordinates, not padded coordinates.
                        offset=np.array([ox,oy]);translation=estimate[:,:2]@offset+estimate[:,2]-offset
                        zoom=float(np.linalg.norm(estimate[0,:2]))
                        if (self.inliers>=10 and self.inliers/good_count>=.6 and .85<=zoom<=1.15
                                and np.linalg.norm(translation)<.25*max(rw,rh)):
                            transform=np.array([[sx,0,ox],[0,sy,oy],[0,0,1.]])
                            full=np.eye(3);full[:2]=estimate
                            warp=(np.linalg.inv(transform)@full@transform)[:2];self.status='applied'
                        else:self.status='rejected_transform'
            except cv2.error:self.status='flow_unavailable'
        self.previous,self.points=gray,points;self.last_warp=warp
        elapsed=(time.perf_counter()-begin)*1000;self.timing={'total_ms':elapsed}
        self.record=dict(ms=elapsed,status=self.status,points=0 if points is None else len(points),
                         inliers=self.inliers,valid_flow=good_count,warp=warp.tolist())
        return warp

def selected_frames(video,entries):
    cap=cv2.VideoCapture(str(video));index=0
    wanted={e['index']:e for e in entries};last=max(wanted)
    try:
        while index<=last:
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Video ended at {index}')
            if index in wanted:
                if frame.shape!=(1080,1920,3):raise ValueError('Native Full HD required')
                yield wanted[index],frame
            index+=1
    finally:cap.release()

def background_mask(detections):
    mask=np.full((1080,1920),255,np.uint8)
    for d in detections:
        a,b,c,e=map(int,d['box']);cv2.rectangle(mask,(max(0,a-15),max(0,b-15)),(min(1919,c+15),min(1079,e+15)),0,-1)
    return mask

def reference_flow(prev,cur,old_dets,dets):
    """Same spatially distributed full-resolution validation points for all variants."""
    g0=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    mask=background_mask(old_dets);points=[]
    for y in range(0,1080,216):
        for x in range(0,1920,240):
            p=cv2.goodFeaturesToTrack(g0[y:y+216,x:x+240],3,.03,25,mask=mask[y:y+216,x:x+240])
            if p is not None:points.extend((p+np.array([x,y])).reshape(-1,2))
    if not points:return [],[]
    p=np.array(points,np.float32).reshape(-1,1,2)
    q,v,err=cv2.calcOpticalFlowPyrLK(g0,g1,p,None,winSize=(31,31),maxLevel=4)
    back,bv,_=cv2.calcOpticalFlowPyrLK(g1,g0,q,None,winSize=(31,31),maxLevel=4)
    xy=q.reshape(-1,2);good=(v.ravel()!=0)&(bv.ravel()!=0)&(np.linalg.norm(back-p,axis=2).ravel()<1.)&(err.ravel()<25)
    good &= (xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
    mask=background_mask(dets);idx=np.clip(xy.astype(int),[0,0],[1919,1079]);good &= mask[idx[:,1],idx[:,0]]>0
    return p.reshape(-1,2)[good].tolist(),xy[good].tolist()

def prepare(recording,out):
    side=[json.loads(x) for x in (recording/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    origin=side[0]['decoded_at'];entries=[]
    for segment,(start,end) in enumerate(WINDOWS):
        next_at=start
        for i,f in enumerate(side):
            t=f['decoded_at']-origin
            if start<=t<=end and t>=next_at:
                d=f.get('overlay',{}).get('decision',{})
                entries.append(dict(index=i,t=t,source_time=f['decoded_at'],frame_id=f['frame_id'],segment=segment,
                                    heading=d.get('heading',{}).get('yaw_deg')))
                next_at+=.1
                if next_at<t:next_at=t+.1
    model,manifest=resolve_model(ROOT,V6_MODEL)
    det=OnnxDroneDetector(model,cpu_threads=2,compute_mode='GPU',diagnostics_dir=out)
    det.settings=replace(det.settings,confidence=.1,iou_threshold=.5)
    previous=None;prior=None;contact=[]
    for n,(entry,frame) in enumerate(selected_frames(recording/'clean.mp4',entries)):
        result=det.detect(frame);entry['detections']=[{k:v for k,v in d.items() if k!='segments'} for d in result['detections']]
        if prior is not None and prior['segment']==entry['segment']:
            entry['reference_before'],entry['reference_after']=reference_flow(previous,frame,prior['detections'],entry['detections'])
        else:entry['reference_before']=[];entry['reference_after']=[]
        if n%60==0:
            thumb=cv2.resize(frame,(480,270));cv2.putText(thumb,f"t={entry['t']:.2f}s yaw={entry['heading']}",(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2);contact.append(thumb)
        if n==60:
            for size in SIZES:
                w,h=size;rw,rh,x,y=geometry(size);canvas=np.full((h,w,3),114,np.uint8)
                canvas[y:y+rh,x:x+rw]=cv2.resize(frame,(rw,rh));cv2.imwrite(str(out/f'canvas_{w}x{h}.png'),canvas)
        previous=frame;prior=entry
        if n%40==0:print(f'CACHE {n+1}/{len(entries)}',flush=True)
    if len(contact)%2:contact.append(np.zeros_like(contact[0]))
    cv2.imwrite(str(out/'contact.jpg'),np.vstack([np.hstack(contact[i:i+2]) for i in range(0,len(contact),2)]))
    metadata=dict(recording=str(recording),source_sha256=hashlib.sha256((recording/'clean.mp4').read_bytes()).hexdigest(),
        source_resolution=[1920,1080],windows=WINDOWS,sample_hz=10,frames=len(entries),model=manifest,
        compute=det.compute_info,python=sys.version,opencv=cv2.__version__,platform=platform.platform(),threads=1,
        fitting='letterbox, no crop or stretch; scale pixel thresholds and LK window from 640-wide baseline',
        scope='Fixed recorded camera path. Fresh cached YOLO on exact sampled frames. BoT-only replay; no Nano/control/flight simulation. Reference flow is NOT ground truth.')
    dump(out/'dataset.json',dict(metadata=metadata,entries=entries))

def verify_baseline(recording,entries):
    live=CameraMotion();candidate=Variant((640,360),400);maximum=0.;count=0
    for e,frame in selected_frames(recording/'clean.mp4',entries[:16]):
        boxes=[d['box'] for d in e['detections'] if d.get('class_id')==0]
        cv2.setRNGSeed(42);a=live.apply(frame,boxes)
        cv2.setRNGSeed(42);b=candidate.apply(frame,boxes)
        maximum=max(maximum,float(np.max(np.abs(a-b))));count+=1
        if not np.allclose(a,b,atol=1e-8) or live.status!=candidate.status:
            raise AssertionError(f'Baseline parity failed: {e["index"]}, {maximum}')
    return dict(frames=count,max_warp_difference=maximum)

def run_variant(recording,entries,size,corners):
    records=[];tracker=None;segment=None;instance=None
    for e,frame in selected_frames(recording/'clean.mp4',entries):
        if e['segment']!=segment:
            BaseTrack._count=0;tracker=BoTSORTTracker((1920,1080));segment=e['segment'];warm=0
            instance=Variant(size,corners)
        ds=[Detection(Box(*d['box']),d['confidence'],d.get('class_id'),d.get('class_name'))
            for d in e['detections'] if d.get('class_id')==0]
        cv2.setRNGSeed(42+e['index'])
        with patch('follow_neo.perception.botsort_motion.CameraMotion',return_value=instance):
            tracker.update(ds,e['source_time'],e['source_time']+.09,e['frame_id'],
                start_confidence=.6,strong_confidence=.6,continuation_confidence=.1,image=frame,
                match_thresholds=(.7,.45,.7))
        r=dict(instance.record,t=e['t'],segment=segment,index=e['index'],heading=e['heading'],
               accepted=tracker.accepted is not None,confirmed=tracker.confirmed,
               has_strong_neo=any(d.confidence>=.6 for d in ds),association=tracker.association_reason,
               identity=tracker.logical_target_id,warmup=warm<3)
        before=np.asarray(e['reference_before']);after=np.asarray(e['reference_after']);r['reference_n']=len(before)
        if len(before)>=12:
            projected=before@instance.last_warp[:,:2].T+instance.last_warp[:,2]
            errors=np.linalg.norm(projected-after,axis=1)
            r.update(residual_median_px=float(np.median(errors)),residual_p95_px=float(np.percentile(errors,95)),
                     identity_residual_median_px=float(np.median(np.linalg.norm(before-after,axis=1))))
        records.append(r);warm+=1
    active=[r for r in records if not r['warmup']]
    return dict(width=size[0],height=size[1],content=list(geometry(size)[:2]),corners=corners,
        timing_ms=stats([r['ms'] for r in active]),points=stats([r['points'] for r in active]),
        status=dict(Counter(r['status'] for r in active)),
        residual_px=stats([r.get('residual_median_px') for r in active]),
        residual_tail_px=stats([r.get('residual_p95_px') for r in active]),
        identity_residual_px=stats([r.get('identity_residual_median_px') for r in active]),
        accepted=sum(r['accepted'] for r in active),frames=len(active),
        strong_neo_frames=sum(r['has_strong_neo'] for r in active),
        strong_neo_rejected=sum(r['has_strong_neo'] and not r['accepted'] for r in active),
        association=dict(Counter(r['association'] for r in active)),records=records)

def report(out,data,results,parity):
    summaries=[{k:v for k,v in r.items() if k!='records'} for r in results]
    dump(out/'summary.json',dict(metadata=data['metadata'],baseline_parity=parity,results=summaries))
    body=[]
    for r in sorted(summaries,key=lambda r:(SIZES.index((r['width'],r['height'])),r['corners'])):
        n=r['frames'];body.append('<tr>'+''.join(f'<td>{v}</td>' for v in [
            f"<a href='points_{r['width']}x{r['height']}_{r['corners']}.png'><bdi>{r['width']}×{r['height']}</bdi></a>",f"<bdi>{r['content'][0]}×{r['content'][1]}</bdi>",r['corners'],
            f"{r['points']['mean']:.0f}",f"{r['timing_ms']['mean']:.1f}",f"{r['timing_ms']['p95']:.1f}",
            f"{100*r['status'].get('applied',0)/n:.1f}%",f"{r['residual_px']['median']:.2f}",
            f"{r['residual_px']['p95']:.2f}",f"{r['accepted']}/{n}",r['strong_neo_rejected']])+'</tr>')
    document='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>השוואת פיצוי תנועת מצלמה</title>
<style>body{font:17px Arial;background:#f2f5f8;color:#172635;max-width:1250px;margin:35px auto;padding:20px}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{border:1px solid #cbd5df;padding:9px;text-align:center}th{background:#17364b;color:white;position:sticky;top:0}h1{font-size:29px}p{line-height:1.7}img{max-width:100%}.note{background:#fff0ca;padding:16px}a{color:#006699}</style>
<h1>24 שילובים — פיצוי תנועת מצלמה</h1>
<p>הקלטת Full HD מקורית, 21 באוקטובר 23:34. קטעים 6–26 ו־32–50 שניות, 337 פריימים; קצב הדגימה בפועל כ־8.6–9.1 לשנייה. אותם פריימים ואותם זיהויי מודל v6 בכל השילובים. חוט OpenCV אחד; שלושה פריימי חימום מוחרגים בכל קטע.</p>
<p class="note">זו בדיקת עקביות על סרטון מוקלט אחד, ולא הוכחה לריקוד חלק יותר. שגיאת הרקע היא מדד מול נקודות בדיקה משותפות, ללא אמת מידה ידנית. השיוך נבדק ב־BoT-SORT בלבד; NanoTrack, בקר הטיסה, הרשת ותגובת הרחפן אינם משוחזרים. זמני המקור קבועים וזהים; זמן החישוב הנמדד אינו מוזן בחזרה לספי התוקף. לכן יתרון קטן בשיוך בתצורה איטית עלול להיעלם בטיסה. אין להמיר את זמן החישוב ל־FPS של המערכת.</p>
<p>כל הגדלים שומרים על התמונה המלאה באמצעות שוליים. ספי פיקסלים, מרחק בין נקודות, גודל שכונת הזיהוי וחלון LK מותאמים לקנה המידה ביחס לרוחב 640. לכן זו השוואת תצורות עם התאמת קנה מידה, ולא החלפת גודל בלבד עם ספי פיקסלים קבועים. המספר 300–450 הוא תקרה; בפועל ייתכנו פחות נקודות. שגיאת הרקע מוצגת בפיקסלים של תמונת המקור: נמוך יותר פירושו התאמה טובה יותר לנקודות הבדיקה. לחיצה על גודל בטבלה פותחת תמונת נקודות של התצורה.</p>
<table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in ['מסגרת','תוכן בפועל','תקרת נקודות','נקודות בפועל בממוצע','זמן ממוצע, מ״ש','זמן באחוזון 95','פיצוי שהתקבל','שגיאת רקע חציונית','אחוזון 95 של שגיאת הרקע','זיהויים ששויכו','זיהויי NEO חזקים שלא שויכו'])+'''</tr></thead><tbody>'''+''.join(body)+'''</tbody></table>
<p>זמני המדידה כוללים את פיצוי המצלמה בלבד. אין בהם פענוח וידאו, YOLO או חישוב מדדי הבדיקה. מספר זיהויים ששויכו גבוה אינו מוכיח שהזהות נכונה; יש לבחון גם איכות והתאמה חזותית.</p>
<p><a href="summary.json">נתונים מלאים</a> · <a href="dataset.json">פריימים, זיהויים ושיטת הבדיקה</a></p>
<h2>תמונות מקטעי הבדיקה</h2><img src="contact.jpg" alt="פריימים מהסרטון עם זמנים וכיוון המצלמה">
<h2>דוגמאות למסגרות המדויקות שנבדקו</h2>'''+''.join(f'<p><bdi>{w}×{h}</bdi></p><img style="max-height:420px" src="canvas_{w}x{h}.png">' for w,h in SIZES)+'''</html>'''
    (out/'report.html').write_text(document,encoding='utf-8')

def finalize(recording,out,data,results):
    if len(results)!=24:raise AssertionError('Incomplete matrix')
    contact=[]
    for e,frame in selected_frames(recording/'clean.mp4',data['entries'][::60]):
        thumb=cv2.resize(frame,(480,270))
        label=f"t={e['t']:.2f}s  yaw={e['heading']:.1f} deg"
        cv2.putText(thumb,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2);contact.append(thumb)
    if len(contact)%2:contact.append(np.zeros_like(contact[0]))
    cv2.imwrite(str(out/'contact.jpg'),np.vstack([np.hstack(contact[i:i+2]) for i in range(0,len(contact),2)]))
    # A common frame demonstrates actual feature selection, outside measured work.
    e,frame=next(selected_frames(recording/'clean.mp4',[data['entries'][60]]))
    for size in SIZES:
        for n in COUNTS:
            v=Variant(size,n);v.apply(frame,[d['box'] for d in e['detections'] if d.get('class_id')==0 and d['confidence']>.6])
            canvas=cv2.cvtColor(v.previous,cv2.COLOR_GRAY2BGR)
            if v.points is not None:
                for x,y in v.points.reshape(-1,2):cv2.circle(canvas,(round(float(x)),round(float(y))),2,(0,255,0),-1)
            cv2.imwrite(str(out/f'points_{size[0]}x{size[1]}_{n}.png'),canvas)
    repeat_path=out/'repeat_checks.json'
    if not repeat_path.exists():
        repeats=[]
        for size,n in [((640,360),400),((960,544),400),((480,270),300)]:
            print('REPEAT',size,n,flush=True)
            r=run_variant(recording,data['entries'],size,n)
            original=next(x for x in results if (x['width'],x['height'],x['corners'])==(*size,n))
            if r['accepted']!=original['accepted'] or r['status']!=original['status']:
                raise AssertionError('Replay quality is not reproducible')
            repeats.append({k:v for k,v in r.items() if k!='records'})
        dump(repeat_path,repeats)
    # Self-contained HTML for saving or sending: examples embedded; full point images remain beside it.
    page=(out/'report.html').read_text(encoding='utf-8')
    repeats=json.loads(repeat_path.read_text(encoding='utf-8'))
    paragraph='<p>בדיקות חזרה נפרדות: '+ ' · '.join(
        f"<bdi>{r['width']}×{r['height']}, {r['corners']}</bdi>: {r['timing_ms']['mean']:.1f} מ״ש בממוצע"
        for r in repeats)+'. תוצאות השיוך וקבלת הפיצוי חזרו באופן זהה. זמני הריצה תלויים בעומס ובמחשב.</p>'
    page=page.replace('<h2>תמונות מקטעי הבדיקה</h2>',paragraph+'<h2>תמונות מקטעי הבדיקה</h2>')
    for name in ['contact.jpg']+[f'canvas_{w}x{h}.png' for w,h in SIZES]:
        mime='image/jpeg' if name.endswith('.jpg') else 'image/png'
        encoded=base64.b64encode((out/name).read_bytes()).decode('ascii')
        page=page.replace('src="'+name+'"','src="data:'+mime+';base64,'+encoded+'"')
    (out/'report.html').write_text(page,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recording',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();recording=args.recording.resolve();out=args.output.resolve()
    if out.is_relative_to(recording):raise ValueError('Never write into source evidence')
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    if not (out/'dataset.json').exists():prepare(recording,out)
    data=json.loads((out/'dataset.json').read_text(encoding='utf-8'));entries=data['entries']
    if data['metadata']['model']['name']!=V6_MODEL:
        raise ValueError('Cached detections use a different model; choose a new output directory')
    if Path(data['metadata']['recording']).resolve()!=recording:
        raise ValueError('Cached dataset belongs to a different recording')
    parity=verify_baseline(recording,entries);print('BASELINE_PARITY',parity,flush=True)
    configs=[(s,n) for s in SIZES for n in COUNTS];random.Random(20261022).shuffle(configs);results=[]
    for i,(size,n) in enumerate(configs):
        path=out/f'{size[0]}x{size[1]}_{n}.json'
        if path.exists():result=json.loads(path.read_text())
        else:
            print(f'RUN {i+1}/24 {size} {n}',flush=True)
            result=run_variant(recording,entries,size,n);dump(path,result)
        results.append(result)
        print('RESULT',size,n,'ms',round(result['timing_ms']['mean'],2),'residual',round(result['residual_px']['median'],3),'accepted',result['accepted'],flush=True)
        report(out,data,results,parity)
    finalize(recording,out,data,results)
    print('DONE',out/'report.html',flush=True)

if __name__=='__main__':main()
