import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Missing component

def analyze_exercise(video_path, exercise_type):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the counter
    counter = ExerciseCounter()

    # Set thresholds for each exercise
    if exercise_type == "pushup":
        down_thresh, up_thresh = 90, 160
    elif exercise_type == "situp":
        down_thresh, up_thresh = 70, 140
    elif exercise_type == "squat":
        down_thresh, up_thresh = 90, 160
    else:
        down_thresh, up_thresh = 90, 160  # default

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0,255,0), thickness=2, circle_radius=2
                    )
                )

                landmarks = results.pose_landmarks.landmark
                angle = process_joints(landmarks, exercise_type)
                if angle is not None:
                    count, feedback = counter.update(angle, down_thresh, up_thresh)
                    cv2.putText(image, f"Reps: {count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.putText(image, feedback, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                feedback = "No person detected"
                cv2.putText(image, feedback, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow('Exercise Analysis', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def process_joints(landmarks, exercise_type):
    # Extract relevant joints
    mp_pose = mp.solutions.pose
    # Left side for simplicity
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    if exercise_type == "pushup":
        angle = calculate_angle(shoulder, elbow, wrist)
        return angle
    elif exercise_type == "situp":
        angle = calculate_angle(knee, hip, shoulder)
        return angle
    elif exercise_type == "squat":
        angle = calculate_angle(hip, knee, ankle)
        return angle
    else:
        return None

    
class ExerciseCounter:
    def __init__(self):
        self.count = 0
        self.state = None  # "up" or "down"

    def update(self, angle, down_thresh, up_thresh):
        feedback = ""
        # Down position detected
        if angle < down_thresh:
            if self.state != "down":
                self.state = "down"
                feedback = "Down position"
        # Up position detected
        elif angle > up_thresh:
            if self.state == "down":
                self.count += 1
                self.state = "up"
                feedback = "Rep counted!"
            else:
                feedback = "Up position"
        else:
            feedback = "In motion"
        return self.count, feedback


def count_reps(current_angle, lower_thresh, upper_thresh):
    counter = ExerciseCounter()
    count = counter.update(current_angle, lower_thresh, upper_thresh)
    return count, "Good form" if 50 < current_angle < 160 else "Adjust form"

if __name__ == "__main__":
    video_path = "C:/Users/Elias/Documents/GitHub/VAoHM/videos/pushup.mp4"  # Replace with your video
    exercise_type = "pushup"  # Change to "situp" or "squat"
    analyze_exercise(video_path, exercise_type)