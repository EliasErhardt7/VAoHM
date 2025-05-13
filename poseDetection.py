import cv2
import mediapipe as mp
import numpy as np
import csv
import joblib
from collections import deque, Counter

clf = joblib.load('exercise_classifier.pkl')
PREDICTION_WINDOW = 15  # Number of frames to consider
pred_queue = deque(maxlen=PREDICTION_WINDOW)

mp_pose = mp.solutions.pose

# IMPORTANT GLOBALS ------------------------------------------------------------

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
# Define relevant landmark indices for each exercise
RELEVANT_LANDMARKS = {
    "pushup": [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ],
    "squat": [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ],
    "situp": [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
    ]
}

# Define relevant connections for each exercise
RELEVANT_CONNECTIONS = {
    "pushup": [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
        (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
        (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    ],
    "squat": [
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    ],
    "situp": [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
        (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    ]
}

# IMPORTANT GLOBALS END ------------------------------------------------------------

# VISUALIZATION FUNCTIONS ------------------------------------------------------------
def draw_relevant_landmarks(image, landmarks, exercise_type):
    # Draw only relevant joints
    for idx in RELEVANT_LANDMARKS[exercise_type]:
        lm = landmarks[idx]
        h, w, _ = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
    # Draw only relevant connections
    for connection in RELEVANT_CONNECTIONS[exercise_type]:
        start_idx, end_idx = connection
        lm_start = landmarks[start_idx]
        lm_end = landmarks[end_idx]
        h, w, _ = image.shape
        x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
        x2, y2 = int(lm_end.x * w), int(lm_end.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 4)

# resize the video to keep the person in frame
def resize_and_pad(image, target_width, target_height):
    h, w = image.shape[:2]
    # Compute scaling factor to fit image inside target box
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create a new image of desired size and fill with black
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # Compute top-left corner for centering
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    # Paste resized image into center of result image
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result

# VISUALIZATION FUNCTIONS END ------------------------------------------------------------

# GET ALL THE FEATURES BEFOREHAND --------------------------------------------------------------

def extract_features(landmarks):
    mp_pose = mp.solutions.pose

    # Helper to get (x, y)
    def get_xy(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    # Left and right joints
    l_shoulder, r_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value), get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    l_elbow, r_elbow = get_xy(mp_pose.PoseLandmark.LEFT_ELBOW.value), get_xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    l_wrist, r_wrist = get_xy(mp_pose.PoseLandmark.LEFT_WRIST.value), get_xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    l_hip, r_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP.value), get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
    l_knee, r_knee = get_xy(mp_pose.PoseLandmark.LEFT_KNEE.value), get_xy(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    l_ankle, r_ankle = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value), get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # Angles
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

    # Feature vector
    features = [
        (left_elbow_angle + right_elbow_angle) / 2,
        (left_knee_angle + right_knee_angle) / 2,
        (left_hip_angle + right_hip_angle) / 2
    ]
    return features

# GET ALL THE FEATURES BEFOREHAND END --------------------------------------------------------------

# MAIN CLASSIFIER FOR REPETITIONS FOR SPECIFIC EXERCISE ------------------------------------------------------

# main function: load video - run pose estimation - draw skeletion onto the video - calculate angles - use angles to count reps
def analyze_exercise(video_path):
    # open video file
    cap = cv2.VideoCapture(video_path)
    # create pose estimator
    mp_pose = mp.solutions.pose
    # to draw the landmarks onto the video
    #mp_drawing = mp.solutions.drawing_utils

    # initialize our counter class
    counter = ExerciseCounter()

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # for mediaPipe each frame needs to be RGB
            frame = resize_and_pad(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # then put frame back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # if a person is detected start pose estimation
            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark)
                pred = clf.predict([features])[0]
                pred_queue.append(pred)
                # Lock in exercise type if there are enough good predictions
                best, count = Counter(pred_queue).most_common(1)[0]
                locked_exercise = None
                if count > PREDICTION_WINDOW // 2:
                    locked_exercise = best
                # Use locked_exercise for all further processing
                if locked_exercise:
                    # Set upper and lower thresholds for each exercise
                    if locked_exercise == "pushup":
                        down_thresh, up_thresh = 90, 160
                    elif locked_exercise == "situp":
                        down_thresh, up_thresh = 40, 80
                    elif locked_exercise == "squat":
                        down_thresh, up_thresh = 110, 160
                    else:
                        down_thresh, up_thresh = 90, 160
                    cv2.putText(
                        image,                     
                        f"Exercise: {locked_exercise.capitalize()}",
                        (670, 50),                    
                        cv2.FONT_HERSHEY_SIMPLEX,   
                        0.8,                       
                        (255, 255, 255),             
                        2,                          
                        cv2.LINE_AA                  
                    )
                    # show only relevant joints
                    draw_relevant_landmarks(image, results.pose_landmarks.landmark, locked_exercise)

                    # get the angles for the Rep counter
                    angle = process_joints(results.pose_landmarks.landmark, locked_exercise)
                    
                    count, feedback = counter.update(angle, down_thresh, up_thresh)
                    # update cv2 Texts
                    cv2.putText(image, f"Reps: {count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, feedback, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                else:
                    feedback = "Detecting exercise..."
                    # update cv2 Texts
                    cv2.putText(image, feedback, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow('Exercise Analysis', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# used to get the angle between 3 joints. For pushups: ( shoulder, elbow, wrist)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# to extract the joint coordinates and compute the angles
def process_joints(landmarks, exercise_type):
    # extract the relevant joints
    mp_pose = mp.solutions.pose
    # Get coordinates for both sides
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    if exercise_type == "pushup":
        left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        angle = (left_angle + right_angle) / 2
        return angle
    elif exercise_type == "situp":
        left_angle = calculate_angle(l_knee, l_hip, l_shoulder)
        right_angle = calculate_angle(r_knee, r_hip, r_shoulder)
        angle = (left_angle + right_angle) / 2
        return angle
    elif exercise_type == "squat":
        left_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_angle = calculate_angle(r_hip, r_knee, r_ankle)
        angle = (left_angle + right_angle) / 2
        return angle
    else:
        return None

'''
# not used
def count_reps(current_angle, lower_thresh, upper_thresh):
    counter = ExerciseCounter()
    count = counter.update(current_angle, lower_thresh, upper_thresh)
    return count, "Good form" if 50 < current_angle < 160 else "Adjust form"
'''

# MAIN CLASSIFIER FOR REPETITIONS FOR SPECIFIC EXERCISE END ------------------------------------------------------

#class to track the number of Reps
class ExerciseCounter:
    def __init__(self):
        self.count = 0
        self.state = None  # "up" or "down"

    def update(self, angle, down_thresh, up_thresh):
        feedback_text = ""
        # state switches to down if the angle is smaller than the down threshold
        if angle < down_thresh:
            if self.state != "down":
                self.state = "down"
                
            feedback_text = "Down position"
        # state switches to down if the angle is bigger than the up threshold
        elif angle > up_thresh:
            if self.state == "down":
                self.count += 1
                self.state = "up"
            feedback_text = "Up position"
        else:
            feedback_text = "In motion"
        return self.count, feedback_text

# runs the script
if __name__ == "__main__":

    video = "C:/Users/Elias/Documents/GitHub/VAoHM/videos/pushup.mp4"
    #locked_exercise = "situp"  # or "squat" or "situp"
    analyze_exercise(video)