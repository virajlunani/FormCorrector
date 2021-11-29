import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
pose = mpPose.Pose()
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mpPose.POSE_CONNECTIONS, 
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape

    cv2.imshow("image", img)

    cv2.waitKey(1)