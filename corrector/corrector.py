import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from drawing_style import *
from check_form import *
import pickle

OUTPUT_CORRECTIONS = True
_BLUE = (255, 0, 0)

def main():
    # load model
    with open('../classifier/model.pkl', 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)

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

                # detect action
                input_vector = np.asarray([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_landmarks]).flatten()

                X = pd.DataFrame([input_vector])
                action = model.predict(X)[0]
                put_text(img, action, (100, 50), _BLUE, 1.25, 2)
                # action_prob = model.predict_proba(X)[0]
                # print(action, action_prob)
                
                if action == 'Curl':
                    back_correct = back_straight(body_landmarks, frame_dims)
                    left_arm_correct, left_elbow_back = left_arm_valid(body_landmarks, frame_dims)
                    right_arm_correct, right_elbow_back = right_arm_valid(body_landmarks, frame_dims)

                    if OUTPUT_CORRECTIONS:
                        if not back_correct:
                            put_text(img, "STRAIGHTEN BACK!", (50, 100))
                        if (not left_arm_correct and left_elbow_back) or (not right_arm_correct and right_elbow_back):
                            put_text(img, "BRING ELBOW FORWARD", (50, 130))
                        if (not left_arm_correct and not left_elbow_back) or (not right_arm_correct and not right_elbow_back):
                            put_text(img, "BRING ELBOW BACK", (50, 130))

                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_curl_drawing_style(back_correct, left_arm_correct, right_arm_correct)
                    )
                elif action == 'Squat':
                    knee_correct = knee_behind_toe(body_landmarks, frame_dims)
                    back_correct = shoulder_behind_knee(body_landmarks, frame_dims)
                    knee_angle = get_knee_angle(body_landmarks, frame_dims)

                    if OUTPUT_CORRECTIONS:
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
                        if not back_correct:
                            put_text(img, "BRING SHOULDERS BACK!", (100, 160))
                    

                    mpDraw.draw_landmarks(
                        img, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=get_squat_drawing_style(knee_correct, back_correct)
                    )

            cv2.imshow("image", img)
            cv2.waitKey(1)
        
    
if __name__ == "__main__":
    main()

