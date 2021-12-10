import math
from mediapipe.python.solutions.pose import PoseLandmark

BACK_ANGLE_THRESHOLD = 5
ARM_ANGLE_THRESHOLD = 7.5


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

    if (abs(math.degrees(math.atan(right_ratio))) < BACK_ANGLE_THRESHOLD or
        abs(math.degrees(math.atan(left_ratio))) < BACK_ANGLE_THRESHOLD):
        return True
    else:
        return False

def left_arm_valid(pose_landmarks, frame_dims):
    width, height = frame_dims
    
    left_shoulder_x = pose_landmarks[PoseLandmark.LEFT_SHOULDER].x * width
    left_shoulder_y = pose_landmarks[PoseLandmark.LEFT_SHOULDER].y * height

    left_elbow_x = pose_landmarks[PoseLandmark.LEFT_ELBOW].x * width
    left_elbow_y = pose_landmarks[PoseLandmark.LEFT_ELBOW].y * height

    left_ratio = float((left_elbow_x - left_shoulder_x) / (left_elbow_y - left_shoulder_y))
    angle = math.degrees(math.atan(left_ratio))
    if abs(angle) < ARM_ANGLE_THRESHOLD:
        return True, None
    else:
        if angle < 0:
            return False, True
        else:
            return False, False

def right_arm_valid(pose_landmarks, frame_dims):
    width, height = frame_dims
    
    right_shoulder_x = pose_landmarks[PoseLandmark.RIGHT_SHOULDER].x * width
    right_shoulder_y = pose_landmarks[PoseLandmark.RIGHT_SHOULDER].y * height

    right_elbow_x = pose_landmarks[PoseLandmark.RIGHT_ELBOW].x * width
    right_elbow_y = pose_landmarks[PoseLandmark.RIGHT_ELBOW].y * height

    right_ratio = float((right_elbow_x - right_shoulder_x) / (right_elbow_y - right_shoulder_y))
    angle = math.degrees(math.atan(right_ratio))
    if abs(angle) < ARM_ANGLE_THRESHOLD:
        return True, None
    else:
        if angle < 0:
            return False, True
        else:
            return False, False
    

def knee_behind_toe(pose_landmarks, frame_dims):
    width, _ = frame_dims
    right_knee_x = pose_landmarks[PoseLandmark.RIGHT_KNEE].x * width

    right_toe_x = pose_landmarks[PoseLandmark.RIGHT_FOOT_INDEX].x * width

    if (right_knee_x > right_toe_x):
        return False
    else:
        return True    

def shoulder_behind_knee(pose_landmarks, frame_dims):
    width, _ = frame_dims

    right_shoulder_x = pose_landmarks[PoseLandmark.RIGHT_SHOULDER].x * width

    right_knee_x = pose_landmarks[PoseLandmark.RIGHT_KNEE].x * width

    if (right_shoulder_x > right_knee_x):
        return False
    else:
        return True    

def get_knee_angle(pose_landmarks, frame_dims):
    width, height = frame_dims

    x1, y1 = pose_landmarks[PoseLandmark.RIGHT_HIP].x * width, pose_landmarks[PoseLandmark.RIGHT_HIP].y * height
    x2, y2 = pose_landmarks[PoseLandmark.RIGHT_KNEE].x * width, pose_landmarks[PoseLandmark.RIGHT_KNEE].y * height
    x3, y3 = pose_landmarks[PoseLandmark.RIGHT_ANKLE].x * width, pose_landmarks[PoseLandmark.RIGHT_ANKLE].y * height

    a = (x2 - x1, y2 - y1)
    b = (x2 - x3, y2 - y3)

    mag_a = math.sqrt(a[0]**2 + a[1]**2)
    mag_b = math.sqrt(b[0]**2 + b[1]**2)

    dot_prod = a[0] * b[0] + a[1] * b[1]

    angle = math.degrees(math.acos(dot_prod / (mag_a * mag_b)))

    return angle

