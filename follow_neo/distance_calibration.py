"""Range from the follower camera to another Mini 4 Pro, for display only."""
import math
import statistics
import numpy as np
from .roi import box_iou

CAMERA_MODEL='DJI Mini 4 Pro'
CAMERA_ZOOM=1
CALIBRATION_CLASS='MINI PRO4'
GLOBAL_LABEL='Mini 4 Pro camera / 1x / other Mini 4 Pro / global'
TARGET_VIEWS={'front':'Front / חזית','rear':'Rear / אחור',
           'left':'Left side / צד שמאל','right':'Right side / צד ימין'}


def view_label(view):
    if view not in TARGET_VIEWS:raise ValueError('Select the visible side of the other Mini 4 Pro')
    return 'Mini 4 Pro camera / 1x / other Mini 4 Pro '+{'front':'front','rear':'rear','left':'left side','right':'right side'}[view]


def profile_view(profile):
    if not profile:return None
    if profile.get('target_view') in TARGET_VIEWS:return profile['target_view']
    # Preserve recognized earlier labels; never silently assign an unknown view.
    label=profile.get('label','').strip().casefold()
    return next((key for key in TARGET_VIEWS if label==view_label(key).casefold()),None)


CAPTURE_SECONDS=10.
MIN_SAMPLES=20
MIN_SPAN_SECONDS=5.


def positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def measured_mini(result,now):
    """Use the detector's full-source box, including when ROI classification is used."""
    if not result or result.get('frame') is None:return None,'Waiting for live detections'
    frame=result['frame']
    if not 0<=now-frame.decoded_at<=.25:return None,'Detection is stale'
    height,width=frame.image.shape[:2]
    candidates=[d for d in result.get('detections',[]) if d.get('model_class_name')==CALIBRATION_CLASS
                and math.isfinite(d.get('confidence',0)) and d.get('confidence',0)>=.5]
    if len(candidates)!=1:return None,'Keep exactly one other Mini 4 Pro visible'
    d=candidates[0]
    # Mini detections are deliberately display-only. Calibration consumes them
    # without changing eligibility or granting the NEO tracker any authority.
    try:
        x1,y1,x2,y2=map(float,d['box'])
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):raise ValueError()
    except (TypeError,ValueError,KeyError):return None,'Invalid detection box'
    w,h=x2-x1,y2-y1
    if w<12 or h<6:return None,'Mini 4 Pro is too small for calibration'
    if x1<=width*.02 or x2>=width*.98 or y1<=height*.02 or y2>=height*.98:
        return None,'Mini 4 Pro box is clipped / near the edge'
    if not (.25<=(x1+x2)/2/width<=.75 and .25<=(y1+y2)/2/height<=.75):
        return None,'Keep the other Mini 4 Pro in the central half of the image'
    via_roi=d.get('roi_status')=='rejected_mini'
    return dict(frame_id=frame.frame_id,time=frame.decoded_at,width_px=w,height_px=h,
                box=[x1,y1,x2,y2],classification_source='roi' if via_roi else 'full_frame',
                resolution=[width,height],roi_id=d.get('roi_id')),(
                    'Mini 4 Pro confirmed by ROI (full-frame width)' if via_roi else 'Mini 4 Pro detected in full frame')


class SampleCapture:
    def __init__(self,distance_m,started,context,kind='calibration'):
        self.distance_m=positive(distance_m,'Known distance (m)')
        self.started=started;self.context=dict(context);self.kind=kind
        self.samples=[];self.last_frame=None;self.last_time=None

    def add(self,measurement):
        if measurement is None:return False
        fid,stamp=measurement['frame_id'],measurement['time']
        if stamp<self.started:return False
        if self.last_frame is not None and (fid<=self.last_frame or stamp<=self.last_time):return False
        if measurement['resolution']!=self.context['resolution']:raise ValueError('Resolution changed; capture again')
        # ROI IDs belong to candidate verification, not aircraft identity: a
        # direct Mini classification clears them. Check the stationary subject
        # against both its first and previous full-frame boxes instead.
        if self.samples and (stamp-self.last_time>.75 or
                any(box_iou(measurement['box'],prior['box'])<.5 for prior in (self.samples[0],self.samples[-1]))):
            raise ValueError('Mini 4 Pro moved or detection was interrupted; keep it stationary and capture again')
        self.last_frame=fid;self.last_time=stamp
        self.samples.append(measurement)
        return True

    def finish(self):
        if len(self.samples)<MIN_SAMPLES:raise ValueError('Not enough valid frames; collect again with the other Mini 4 Pro centered')
        span=self.samples[-1]['time']-self.samples[0]['time']
        if span<MIN_SPAN_SECONDS:raise ValueError('Valid observations must span at least 5 seconds')
        widths=[s['width_px'] for s in self.samples]
        width=statistics.median(widths)
        spread=float(np.percentile(widths,90)-np.percentile(widths,10))/width
        if spread>.20:raise ValueError('Box size changed too much; hold distance and Mini 4 Pro orientation fixed')
        return dict(distance_m=self.distance_m,width_px=width,samples=len(widths),
                    span_seconds=span,relative_spread=spread)


def _clean_points(points,context,max_points=400):
    if not isinstance(context,dict) or not isinstance(context.get('resolution'),list) or len(context['resolution'])!=2:
        raise ValueError('Missing camera resolution')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in context['resolution']):raise ValueError('Invalid resolution')
    if not isinstance(context.get('model'),str) or not context['model']:raise ValueError('Missing detection model')
    if not isinstance(points,list) or not 3<=len(points)<=max_points:raise ValueError(f'Collect at least 3 points, up to {max_points}')
    cleaned=[]
    for point in points:
        if not isinstance(point,dict):raise ValueError('Invalid calibration sample')
        d=positive(point.get('distance_m'),'Distance');w=positive(point.get('width_px'),'Box width')
        samples=point.get('samples',0);span=positive(point.get('span_seconds'),'Sample span')
        spread=point.get('relative_spread',0)
        if (not isinstance(samples,int) or isinstance(samples,bool) or samples<MIN_SAMPLES or span<MIN_SPAN_SECONDS
                or not isinstance(spread,(int,float)) or not math.isfinite(spread) or not 0<=spread<=.20
                or w>context['resolution'][0]):raise ValueError('Invalid or unstable calibration sample')
        cleaned.append(dict(distance_m=d,width_px=w,samples=samples,span_seconds=span,relative_spread=spread))
    distances=[p['distance_m'] for p in cleaned]
    if len({round(d,3) for d in distances})<3 or max(distances)/min(distances)<1.5:
        raise ValueError('Use 3 different distances spanning at least a 1.5x range')
    return cleaned


def fit_profile(points,context,label,*,target_view=None):
    """Validate the previous single-side format for migration."""
    if target_view is not None:label=view_label(target_view)
    if not isinstance(label,str) or not label.strip():raise ValueError('Enter a camera / zoom / Mini 4 Pro view label')
    cleaned=_clean_points(points,context,max_points=100)
    distances=[p['distance_m'] for p in cleaned]
    # Equal weight per station; longer captures must not dominate the fit.
    k=statistics.median(p['distance_m']*p['width_px'] for p in cleaned)
    errors=[abs(k/p['width_px']-p['distance_m'])/p['distance_m'] for p in cleaned]
    if max(errors)>.25:raise ValueError('Calibration is inconsistent (>25% fit error); check distances, zoom and Mini 4 Pro view')
    profile=dict(schema_version=2,target_class=CALIBRATION_CLASS,camera_model=CAMERA_MODEL,zoom=CAMERA_ZOOM,
                label=label.strip()[:160],context=dict(context),points=cleaned,k_m_px=k,
                min_distance_m=min(distances),max_distance_m=max(distances),
                fit_median_error=float(statistics.median(errors)),fit_max_error=max(errors),validations=[])
    if target_view is not None:profile['target_view']=target_view
    return profile


def load_profile(data):
    if data is None:return None
    if (not isinstance(data,dict) or data.get('schema_version')!=2
            or data.get('target_class')!=CALIBRATION_CLASS):
        raise ValueError('Saved calibration is not for the other Mini 4 Pro; collect a new calibration')
    view=data.get('target_view')
    if data.get('camera_model')!=CAMERA_MODEL or isinstance(data.get('zoom'),bool) or data.get('zoom')!=CAMERA_ZOOM:
        raise ValueError('Calibration requires Mini 4 Pro at 1x')
    profile=fit_profile(data.get('points'),data.get('context'),data.get('label'),target_view=view)
    # Recompute validation errors too; never trust stored fit coefficients.
    records=data.get('validations',[])
    if not isinstance(records,list):raise ValueError('Invalid validation history')
    for record in records[-20:]:
        if not isinstance(record,dict):raise ValueError('Invalid validation sample')
        d=positive(record.get('distance_m'),'Validation distance');w=positive(record.get('width_px'),'Validation width')
        estimate=profile['k_m_px']/w
        profile['validations'].append(dict(distance_m=d,width_px=w,estimated_m=estimate,error_m=estimate-d,relative_error=abs(estimate-d)/d))
    return profile


def estimate_distance(profile,measurement,context):
    if profile is None:return None,'No saved calibration'
    if profile.get('schema_version') not in (2,4) or profile.get('target_class')!=CALIBRATION_CLASS:
        return None,'Calibration is not for the other Mini 4 Pro'
    if measurement is None:return None,'No valid measurement'
    if context!=profile['context']:return None,'Camera source, model or resolution differs from calibration'
    distance=profile['k_m_px']/measurement['width_px']
    inside=profile['min_distance_m']<=distance<=profile['max_distance_m']
    quality='Within calibrated range' if inside else 'Outside calibrated range (extrapolated)'
    if profile.get('schema_version')==4:
        low=profile['k_min_m_px']/measurement['width_px']
        high=profile['k_max_m_px']/measurement['width_px']
        quality=f'Observed calibration spread: {low:.2f}-{high:.2f} m | '+quality
    return distance,quality


def load_calibrations(data):
    """Load independent side profiles, including the previous single-side format."""
    if data is None:
        return dict(schema_version=3,target_class=CALIBRATION_CLASS,selected_view='front',profiles={})
    if not isinstance(data,dict):raise ValueError('Invalid distance calibrations')
    if data.get('schema_version')==2:
        profile=load_profile(data)
        view=profile_view(profile)
        if view is None:raise ValueError('Saved Mini 4 Pro calibration has no recognized side')
        # Preserve the measured points and validation history; never reuse one
        # side's coefficient for another side with a different apparent width.
        profile['target_view']=view
        return dict(schema_version=3,target_class=CALIBRATION_CLASS,selected_view=view,profiles={view:profile})
    if data.get('schema_version')!=3 or data.get('target_class')!=CALIBRATION_CLASS:
        raise ValueError('Saved calibration is not for the other Mini 4 Pro; collect a new calibration')
    view=data.get('selected_view')
    if view not in TARGET_VIEWS:raise ValueError('Invalid selected Mini 4 Pro side')
    profiles=data.get('profiles')
    if not isinstance(profiles,dict) or any(key not in TARGET_VIEWS for key in profiles):
        raise ValueError('Invalid Mini 4 Pro calibration sides')
    cleaned={}
    for side,raw in profiles.items():
        profile=load_profile(raw)
        if profile_view(profile)!=side:raise ValueError('Saved calibration side does not match its profile')
        cleaned[side]=profile
    return dict(schema_version=3,target_class=CALIBRATION_CLASS,selected_view=view,profiles=cleaned)


def fit_global_profile(points,context):
    """One inverse-width estimate pooled across orientations, with observed spread.

    Each known distance gets equal weight, even when it has more captures or
    longer recordings. The min/max envelope is empirical, not a confidence
    interval or a guarantee for unseen orientations.
    """
    cleaned=_clean_points(points,context)
    stations={}
    products=[]
    for point in cleaned:
        value=point['distance_m']*point['width_px']
        products.append(value)
        stations.setdefault(round(point['distance_m'],3),[]).append(value)
    k=statistics.median(statistics.median(values) for values in stations.values())
    errors=[abs(k/p['width_px']-p['distance_m'])/p['distance_m'] for p in cleaned]
    distances=[p['distance_m'] for p in cleaned]
    # Unlike the old side-specific fit, variation across fixed-view captures
    # is expected here. Keep it visible rather than rejecting other directions.
    return dict(schema_version=4,mode='global',target_class=CALIBRATION_CLASS,
                camera_model=CAMERA_MODEL,zoom=CAMERA_ZOOM,label=GLOBAL_LABEL,
                context=dict(context),points=cleaned,k_m_px=k,
                k_min_m_px=min(products),k_max_m_px=max(products),
                min_distance_m=min(distances),max_distance_m=max(distances),
                fit_median_error=float(statistics.median(errors)),fit_max_error=max(errors),validations=[])


def load_global_profile(data):
    """Recompute a global profile, pooling compatible saved side calibrations."""
    if data is None:return None
    if not isinstance(data,dict) or data.get('target_class')!=CALIBRATION_CLASS:
        raise ValueError('Saved calibration is not for the other Mini 4 Pro; collect a new calibration')
    version=data.get('schema_version')
    if version==4:
        if (data.get('mode')!='global' or data.get('camera_model')!=CAMERA_MODEL
                or isinstance(data.get('zoom'),bool) or data.get('zoom')!=CAMERA_ZOOM):
            raise ValueError('Global calibration requires Mini 4 Pro at 1x')
        profile=fit_global_profile(data.get('points'),data.get('context'))
        records=data.get('validations',[])
    elif version in (2,3):
        saved=[load_profile(data)] if version==2 else list(load_calibrations(data)['profiles'].values())
        if not saved:return None
        context=saved[0]['context']
        if any(p['context']!=context for p in saved):
            raise ValueError('Saved sides use different camera sources, models or resolutions; they cannot be pooled')
        profile=fit_global_profile([point for p in saved for point in p['points']],context)
        records=[record for p in saved for record in p['validations']]
    else:
        raise ValueError('Unsupported distance calibration')
    if not isinstance(records,list):raise ValueError('Invalid validation history')
    for record in records:
        if not isinstance(record,dict):raise ValueError('Invalid validation sample')
        distance=positive(record.get('distance_m'),'Validation distance')
        width=positive(record.get('width_px'),'Validation width')
        estimate=profile['k_m_px']/width
        profile['validations'].append(dict(distance_m=distance,width_px=width,estimated_m=estimate,
            error_m=estimate-distance,relative_error=abs(estimate-distance)/distance))
    return profile
