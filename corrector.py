import cv2
import mediapipe as mp
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            _, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS
                )
                
                body_landmarks = results.pose_landmarks.landmark
                print(body_landmarks)


            cv2.imshow("image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()

