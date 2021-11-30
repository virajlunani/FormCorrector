from typing import Tuple, Mapping
from mediapipe.python.solutions import pose

from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_WHITE = (224, 224, 224)

_THICK =  3
_THIN = 2

BACK_LANDMARKS = frozenset([
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_HIP
])

def get_back_correct_drawing_style() -> Mapping[Tuple[int, int], DrawingSpec]:
    pose_landmark_style = {}
    back_connections = [(11, 12), (11, 23), (23, 24), (12, 24)]
    back_spec = DrawingSpec(_GREEN, thickness=_THICK)
    body_spec = DrawingSpec(_WHITE, thickness=_THIN)

    for connection in POSE_CONNECTIONS:
        if connection in back_connections:
            pose_landmark_style[connection] = back_spec
        else:
            pose_landmark_style[connection] = body_spec

    return pose_landmark_style

def get_back_incorrect_drawing_style() -> Mapping[Tuple[int, int], DrawingSpec]:
    pose_landmark_style = {}
    back_connections = [(11, 12), (11, 23), (23, 24), (12, 24)]
    back_spec = DrawingSpec(_RED, thickness=_THICK)
    body_spec = DrawingSpec(_WHITE, thickness=_THIN)

    for connection in POSE_CONNECTIONS:
        if connection in back_connections:
            pose_landmark_style[connection] = back_spec
        else:
            pose_landmark_style[connection] = body_spec

    return pose_landmark_style

