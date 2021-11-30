import math
import cv2
import mediapipe as mp
import numpy as np
from drawing_style import get_back_correct_drawing_style, get_back_incorrect_drawing_style
from check_form import back_straight

def main():
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, img = cap.read()
            frame_dims = img.shape[1::-1]
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)

            if results.pose_landmarks:
                body_landmarks = results.pose_landmarks.landmark
                if back_straight(body_landmarks, frame_dims):
                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_back_correct_drawing_style()
                    )
                else:
                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_back_incorrect_drawing_style()
                    )
                
                

            cv2.imshow("image", img)
            cv2.waitKey(1)
def test_tan():
    x1, y1 = 40, 10
    x2, y2 = 30, 90
    print(math.degrees(math.atan(float((x2-x1)/(y2-y1)))))
    
if __name__ == "__main__":
    main()

