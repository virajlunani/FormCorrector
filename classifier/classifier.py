import cv2
import mediapipe as mp
import numpy as np
from csv_helpers import write_headers, write_vector

def generate_training_data(label):
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)

            if results.pose_landmarks and results.face_landmarks:
                mpDraw.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS
                )
                mpDraw.draw_landmarks(
                    img, 
                    results.face_landmarks, 
                    mp_holistic.FACEMESH_CONTOURS
                )
                body_landmarks = results.pose_landmarks.landmark
                face_landmarks = results.face_landmarks.landmark
                

                body_landmarks_list = np.asarray([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_landmarks]).flatten()
                face_landmarks_list = np.asarray([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face_landmarks]).flatten()

                feature_vector = np.concatenate([body_landmarks_list, face_landmarks_list])

                write_vector(label, feature_vector)

            cv2.imshow("image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    generate_training_data("Sad")


