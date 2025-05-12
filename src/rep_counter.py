import cv2
import numpy as np
from collections import deque
import pyopenpose as op

# === OpenPose init ===
params = {"model_folder": "C:/Users/Elias/Documents/GitHub/humanAnalysis/openpose/models",
          "model_pose"          : "BODY_25",
            "net_resolution"      : "656x368",}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 2) Utility: compute angle between three points
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# 3) StateMachine class to count reps
class RepCounter:
    def __init__(self, joint_idxs, down_thresh, up_thresh):
        self.joint_idxs = joint_idxs      # tuple of (a, b, c) indices in OpenPose output
        self.down_thresh = down_thresh    # angle when “down” position
        self.up_thresh = up_thresh        # angle when “up” position
        self.state = "up"                 # current state: "up" or "down"
        self.count = 0

    def update(self, keypoints):
        # Extract 3D coords of the three joints
        pts = [keypoints[p] for p in self.joint_idxs]
        ang = angle(np.array(pts[0]), np.array(pts[1]), np.array(pts[2]))
        # State machine
        if self.state == "up" and ang < self.down_thresh:
            self.state = "down"
        elif self.state == "down" and ang > self.up_thresh:
            self.state = "up"
            self.count += 1

# 4) Configure counters for each exercise
# OpenPose BODY_25 joint indices: e.g., elbow: 2=RShoulder,3=RElbow,4=RWrist
counters = {
    "pushup": RepCounter(joint_idxs=(2,3,4), down_thresh=60, up_thresh=160),
    "situp":  RepCounter(joint_idxs=(8,1,0), down_thresh=90, up_thresh=150),
    "squat":  RepCounter(joint_idxs=(8,9,10), down_thresh=70, up_thresh=160),
}

# 5) Process video
cap = cv2.VideoCapture("your_video.mp4")
frame_history = deque(maxlen=5)  # optional smoothing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    kps = datum.poseKeypoints

    if kps is None: continue
    human = kps[0][:,:2]  # take first detected person

    # Update each counter
    for name, rc in counters.items():
        rc.update(human)

    # Display counts
    out = datum.cvOutputData
    y = 30
    for name, rc in counters.items():
        cv2.putText(out, f"{name}: {rc.count}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        y += 30

    cv2.imshow("Reps", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()