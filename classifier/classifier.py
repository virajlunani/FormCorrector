import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from csv_helpers import write_headers, write_vector

def generate_class_data(label, video):
    cap = cv2.VideoCapture(video)
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
                

                feature_vector = np.asarray([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in body_landmarks]).flatten()

                write_vector(label, feature_vector)

            cv2.imshow("image", img)
            cv2.waitKey(1)

def train(train_data):
    df = pd.read_csv(train_data)
    X = df.drop("class", axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
    
    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))

    with open('model.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)


def generate_training_data():
    write_headers()
    generate_class_data("Curl", "training_videos/curl.mp4")
    generate_class_data("Squat", "training_videos/squat.mp4")
    
if __name__ == "__main__":
    train('train_data.csv')