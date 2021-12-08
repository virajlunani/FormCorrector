import sys
import cv2
import mediapipe as mp
import numpy as np
from drawing_style import *
from check_form import *

def main(argv):
    cap = cv2.VideoCapture('squat.mp4')
    mpDraw = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    stage = None
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, img = cap.read()
            
            frame_dims = img.shape[1::-1]
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)

            if results.pose_landmarks:
                body_landmarks = results.pose_landmarks.landmark
                if argv[0] == 'curl':
                    back_correct = back_straight(body_landmarks, frame_dims)
                    left_arm_correct = left_arm_valid(body_landmarks, frame_dims)
                    right_arm_correct = right_arm_valid(body_landmarks, frame_dims)
                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_curl_drawing_style(back_correct, left_arm_correct, right_arm_correct)
                    )
                    '''
                    if back not straight --> display text on screen "STRAIGHTEN BACK"
                    if elbow too far forward display text on screen "BRING ELBOW BACK"
                    if elbow too far back display text on screen "BRING ELBOW FORWARD"
                    '''
                elif argv[0] == 'squat':
                    knee_correct = knee_behind_toe(body_landmarks, frame_dims)
                    knee_angle = get_knee_angle(body_landmarks, frame_dims)

                    
                    if knee_angle > 160:
                        stage = "up"
                    if knee_angle > 100 and stage == "up":
                        put_text(img, "SQUAT LOWER", (100, 100))
                    if knee_angle < 100:
                        stage = "down"
                        put_text(img, "GOOD JOB!", (100, 100))
                    if knee_angle > 100 and knee_angle < 120 and stage == "down":
                        put_text(img, "GOOD JOB!", (100, 100))
                    if not knee_correct:
                        put_text(img, "BRING KNEES BACK!", (100, 130))
                    

                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_squat_drawing_style(knee_correct)
                    )                    

            cv2.imshow("image", img)
            cv2.waitKey(1)
    
if __name__ == "__main__":
    main(sys.argv[1:])

