import cv2
import mediapipe as mp
import numpy as np
import csv
import joblib

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

def save_features_from_video(video_path, label, output_csv):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose, open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark)
                writer.writerow(features + [label])
    cap.release()

# GET ALL THE FEATURES BEFOREHAND END --------------------------------------------------------------

# runs the script
if __name__ == "__main__":
    output_csv = "features.csv"

    # Pushup videos
    save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/pushup.mp4", "pushup", output_csv)
    #save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/Push-Ups_Train.mp4", "pushup", output_csv)
    #save_features_from_video("pushup2.mp4", "pushup", output_csv)

    # Squat videos
    save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/squat.mp4", "squat", output_csv) 
    #save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/Squat_Train.mp4", "squat", output_csv) 
    #save_features_from_video("squat2.mp4", "squat", output_csv)

    # Situp videos
    #save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/Sit-Ups_Train1.mp4", "situp", output_csv) 
    save_features_from_video("C:/Users/Elias/Documents/GitHub/VAoHM/videos/Sit-Ups_Train2.mp4", "situp", output_csv)
    #save_features_from_video("situp2.mp4", "situp", output_csv)

    #analyze_exercise(video)