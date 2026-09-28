"""Bounded sparse-flow camera motion estimation for BoT-SORT association."""
import cv2
import numpy as np


class CameraMotion:
    def __init__(self):
        self.previous = None
        self.points = None
        self.status = 'initializing'
        self.inliers = 0
        self.last_warp = np.eye(2, 3, dtype=np.float64)

    def apply(self, image, detections=None):
        identity = np.eye(2, 3, dtype=np.float64)
        self.last_warp = identity
        self.inliers = 0
        if image is None:
            self.previous = self.points = None
            self.status = 'image_unavailable'
            return identity
        height, width = image.shape[:2]
        scale = min(1., 640. / width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (max(1, round(width*scale)), max(1, round(height*scale))))
        sx, sy = gray.shape[1]/width, gray.shape[0]/height
        mask = np.full(gray.shape, 255, np.uint8)
        if detections is not None:
            for box in detections:
                x1, y1, x2, y2 = np.asarray(box[:4]) * [sx, sy, sx, sy]
                cv2.rectangle(mask, (max(0, int(x1)-3), max(0, int(y1)-3)),
                              (min(gray.shape[1]-1, int(x2)+3), min(gray.shape[0]-1, int(y2)+3)), 0, -1)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=.01,
                                       minDistance=7, mask=mask)
        warp = identity
        self.status = 'insufficient_features'
        if self.previous is not None and self.previous.shape == gray.shape and self.points is not None and len(self.points) >= 12:
            try:
                current, valid, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None)
                back, backward_valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, current, None)
                good = ((valid.ravel() != 0) & (backward_valid.ravel() != 0)
                        & (np.linalg.norm(back-self.points, axis=2).ravel() < 1.5))
                before, after = self.points[good], current[good]
                if len(before) >= 12:
                    estimate, inliers = cv2.estimateAffinePartial2D(
                        before, after, method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=500)
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
        self.previous, self.points = gray, points
        self.last_warp = warp
        return warp
