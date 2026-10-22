import numpy as np
import cv2

class FrameTracker:
    def __init__(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None

    def initialize_keyframe(self, image):
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

    def compute_disparity(self, image, min_disparity):
        if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < 10:
            self.initialize_keyframe(image)
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_kf = self.kf_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_kf) < 10:
            self.initialize_keyframe(image)
            return True

        # Measure displacement from keyframe to current frame
        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disparity = np.mean(displacement)

        if mean_disparity > min_disparity:
            self.initialize_keyframe(image)
            return True
        else:
            return False
