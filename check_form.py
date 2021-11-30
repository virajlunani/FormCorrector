import math
from mediapipe.python.solutions.pose import PoseLandmark

def back_straight(pose_landmarks, frame_dims):
    width, height = frame_dims
    right_hip_x = pose_landmarks[PoseLandmark.RIGHT_HIP].x * width
    right_hip_y = pose_landmarks[PoseLandmark.RIGHT_HIP].y * height
    
    right_shoulder_x = pose_landmarks[PoseLandmark.RIGHT_SHOULDER].x * width
    right_shoulder_y = pose_landmarks[PoseLandmark.RIGHT_SHOULDER].y * height

    left_hip_x = pose_landmarks[PoseLandmark.LEFT_HIP].x * width
    left_hip_y = pose_landmarks[PoseLandmark.LEFT_HIP].y * height
    
    left_shoulder_x = pose_landmarks[PoseLandmark.LEFT_SHOULDER].x * width
    left_shoulder_y = pose_landmarks[PoseLandmark.LEFT_SHOULDER].y * height

    right_ratio = float((right_hip_x - right_shoulder_x) / (right_hip_y - right_shoulder_y))
    left_ratio = float((left_hip_x - left_shoulder_x) / (left_hip_y - left_shoulder_y))

    if (abs(math.degrees(math.atan(right_ratio))) < 5 or
        abs(math.degrees(math.atan(left_ratio))) < 5):
        return True
    else:
        return False
    



