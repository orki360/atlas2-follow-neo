"""Bounded sparse-flow camera motion estimation for BoT-SORT association."""
from follow_neo.diagnostics import activity
import cv2
import numpy as np
import time
from follow_neo.shared.gmc import GMC_DEFAULT_RESOLUTION, GMC_DEFAULT_CORNERS, validate_gmc


def _odd(value, minimum=3):
    n = max(minimum, round(value))
    return n if n % 2 else n + 1


class CameraMotion:
    def __init__(self, resolution=GMC_DEFAULT_RESOLUTION, max_corners=GMC_DEFAULT_CORNERS):
        validate_gmc(resolution, max_corners)
        self.resolution, self.max_corners = resolution, max_corners
        self.bounds = tuple(map(int, resolution.split('x')))
        self._reset_pair()

    def _reset_pair(self):
        self.previous = None
        self.points = None
        self.source_size = None
        self.working_size = None
        self.status = 'initializing'
        self.inliers = 0
        self.last_warp = np.eye(2, 3, dtype=np.float64)
        self.timing = {}

    def configure(self, resolution, max_corners):
        validate_gmc(resolution, max_corners)
        if (resolution, max_corners) == (self.resolution, self.max_corners):
            return
        self.resolution, self.max_corners = resolution, max_corners
        self.bounds = tuple(map(int, resolution.split('x')))
        # Never compare different geometries or reuse points from the old budget.
        # The surrounding BoT track identity and Kalman state remain intact.
        self._reset_pair()

    def apply(self, image, detections=None):
        timing_token = activity.token()
        started = activity.clock()
        self.timing = {}
        identity = np.eye(2, 3, dtype=np.float64)
        self.last_warp = identity
        self.inliers = 0
        if image is None:
            self.previous = self.points = None
            self.status = 'image_unavailable'
            elapsed = activity.duration_ms(timing_token, started)
            if elapsed is not None:self.timing['total_ms'] = elapsed
            return identity
        height, width = image.shape[:2]
        if self.source_size != (width, height):
            self.previous = self.points = None
        self.source_size = (width, height)
        # The 640 preset retains the original path, including non-16:9 input.
        # Other presets fit within their bounds without cropping or stretching.
        scale = min(1., 640. / width) if self.resolution == '640x360' else min(
            1., self.bounds[0] / width, self.bounds[1] / height)
        small = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.working_size = (gray.shape[1], gray.shape[0])
        sx, sy = gray.shape[1]/width, gray.shape[0]/height
        # Scale pixel tolerances with working resolution, keeping the original
        # physical gates. At 640 every parameter is exactly unchanged.
        pixel_scale = gray.shape[1] / min(width, 640)
        margin = max(1, round(3 * pixel_scale))
        prepared = activity.clock()
        if activity.enabled:self.timing['preprocess_ms'] = (prepared-started)*1000
        mask = np.full(gray.shape, 255, np.uint8)
        if detections is not None:
            for box in detections:
                x1, y1, x2, y2 = np.asarray(box[:4]) * [sx, sy, sx, sy]
                cv2.rectangle(mask, (max(0, int(x1)-margin), max(0, int(y1)-margin)),
                              (min(gray.shape[1]-1, int(x2)+margin), min(gray.shape[0]-1, int(y2)+margin)), 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, qualityLevel=.01,
                                       minDistance=max(1., 7*pixel_scale), mask=mask,
                                       blockSize=_odd(3*pixel_scale))
        featured = activity.clock()
        if activity.enabled:self.timing['features_ms'] = (featured-prepared)*1000
        warp = identity
        self.status = 'insufficient_features'
        if self.previous is not None and self.previous.shape == gray.shape and self.points is not None and len(self.points) >= 12:
            try:
                window = _odd(21*pixel_scale, 5)
                current, valid, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, winSize=(window, window))
                back, backward_valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, current, None, winSize=(window, window))
                good = ((valid.ravel() != 0) & (backward_valid.ravel() != 0)
                        & (np.linalg.norm(back-self.points, axis=2).ravel() < 1.5*pixel_scale))
                flowed = activity.clock()
                if activity.enabled:self.timing['flow_ms'] = (flowed-featured)*1000
                before, after = self.points[good], current[good]
                if len(before) >= 12:
                    estimate, inliers = cv2.estimateAffinePartial2D(
                        before, after, method=cv2.RANSAC, ransacReprojThreshold=2.*pixel_scale, maxIters=500)
                    self.inliers = 0 if inliers is None else int(inliers.sum())
                    if estimate is not None and np.isfinite(estimate).all():
                        zoom = float(np.linalg.norm(estimate[0, :2]))
                        if self.inliers >= 10 and self.inliers/len(before) >= .6 and .85 <= zoom <= 1.15 and np.linalg.norm(estimate[:, 2]) < .25*max(gray.shape):
                            transform = np.diag([sx, sy, 1.])
                            full = np.eye(3); full[:2] = estimate
                            warp = (np.linalg.inv(transform) @ full @ transform)[:2]
                            self.status = 'applied'
                        else:
                            self.status = 'rejected_transform'
            except cv2.error:
                self.status = 'flow_unavailable'
            if activity.enabled:self.timing['estimate_ms'] = (activity.clock()-featured)*1000-self.timing.get('flow_ms',0.)
        self.previous, self.points = gray, points
        self.last_warp = warp
        if activity.enabled:self.timing['total_ms'] = (activity.clock()-started)*1000
        if not activity.valid(timing_token):self.timing = {}
        return warp
