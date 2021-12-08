from typing import Tuple, Mapping
from mediapipe.python.solutions import pose

from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
import cv2

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_WHITE = (224, 224, 224)
_YELLOW = (0, 255, 255)

_THICK =  3
_THIN = 2

BACK_LANDMARKS = frozenset([
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_HIP
])

BACK_CONNECTIONS = [(11, 12), (11, 23), (23, 24), (12, 24)]
LEFT_ARM_CONNECTIONS = [(11, 13), (13, 15)]
RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16)]
RIGHT_LEG_CONNECTION = (26,28)

CORRECT_SPEC = DrawingSpec(_GREEN, thickness=_THICK)
INCORRECT_SPEC = DrawingSpec(_RED, thickness=_THICK)
BODY_SPEC = DrawingSpec(_WHITE, thickness=_THIN)

def get_curl_drawing_style(back_correct, left_arm_correct, right_arm_correct) -> Mapping[Tuple[int, int], DrawingSpec]:
    pose_landmark_style = {}

    for connection in POSE_CONNECTIONS:
        if connection in BACK_CONNECTIONS:
            if back_correct:
                pose_landmark_style[connection] = CORRECT_SPEC
            else:
                pose_landmark_style[connection] = INCORRECT_SPEC
        elif connection in LEFT_ARM_CONNECTIONS:
            if left_arm_correct:
                pose_landmark_style[connection] = CORRECT_SPEC
            else:
                pose_landmark_style[connection] = INCORRECT_SPEC
        elif connection in RIGHT_ARM_CONNECTIONS:
            if right_arm_correct:
                pose_landmark_style[connection] = CORRECT_SPEC
            else:
                pose_landmark_style[connection] = INCORRECT_SPEC
        else:
            pose_landmark_style[connection] = BODY_SPEC

    return pose_landmark_style

def get_squat_drawing_style(knee_correct) -> Mapping[Tuple[int, int], DrawingSpec]:
    pose_landmark_style = {}

    for connection in POSE_CONNECTIONS:
        if connection == RIGHT_LEG_CONNECTION:
            if knee_correct:
                pose_landmark_style[connection] = CORRECT_SPEC
            else:
                pose_landmark_style[connection] = INCORRECT_SPEC
        else:
            pose_landmark_style[connection] = BODY_SPEC

    return pose_landmark_style

def put_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, _YELLOW, 2, cv2.LINE_AA)
