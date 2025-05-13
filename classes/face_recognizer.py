import cv2
import joblib
import numpy as np
import os
import random
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
class Face_recognizer():
    def __init__(self, output_recognition_viewer, output_detection_viewer):
        self.output_recognition_viewer = output_recognition_viewer
        self.output_detection_viewer = output_detection_viewer
        self.pca_features = None
        self.model = joblib.load("classes\svm_pca_model.pkl")
        self.threshold = 0.2
        self.output_label = None
        self.train_data_dir = "data\ORL database"
        self.outside_img = cv2.imread("data\staff_only.jpg")
        self.recognition_label = ""
        # alarm not in data
        self.alarmPlayer = QMediaPlayer()
        alarm_sound_url = QUrl.fromLocalFile("Access Denied.mp3")
        self.alarmPlayer.setMedia(QMediaContent(alarm_sound_url))
        self.alarmPlaying = False
        # accepted
        self.acceptedPlayer = QMediaPlayer()
        accepted_sound_url = QUrl.fromLocalFile("accept sound effect.mp3")
        self.acceptedPlayer.setMedia(QMediaContent(accepted_sound_url))
        self.acceptedPlaying = False

    def apply_face_recognition(self):
        # print(self.output_detection_viewer)
        self.pca_features = self.output_detection_viewer.current_image.pca_features
        print(self.pca_features.shape)
        probs = self.model.predict_proba(self.pca_features.reshape(1, -1))
        print(probs)
        self.max_prob = np.max(probs)
        predicted_label = self.model.predict(self.pca_features.reshape(1, -1))[0]
        self.output_label = predicted_label
        if self.max_prob >= self.threshold:
            # predicted_label = self.model.predict(self.pca_features.reshape(1, -1))[0]
            # self.output_label = predicted_label
            print(f"The output label is {self.output_label}")
            # set the recognition label:
            self.recognition_label = f"This Image Matches person with ID :{self.output_label}"
            label_folder = os.path.join(self.train_data_dir, str(self.output_label))
            if os.path.exists(label_folder):
                image_files = [f for f in os.listdir(label_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    random_image_file = random.choice(image_files)
                    image_path = os.path.join(label_folder, random_image_file)
                    img = cv2.imread(image_path)
                    if img is not None:
                        # img = cv2.resize(img, (self.output_recognition_viewer.width, self.output_recognition_viewer.height))
                        # set image
                        self.output_recognition_viewer.setImage(cv2.transpose(img))
                        print("done setting image")
                        self.acceptedPlayer.play()
                        self.acceptedPlaying = True
        else :
            print("outtttt")
            img_bgr = cv2.cvtColor(self.outside_img, cv2.COLOR_RGB2BGR)
            self.output_recognition_viewer.setImage(cv2.transpose(img_bgr))
            self.recognition_label = f"This Image doesn't Exist, but Matches person with ID: {self.output_label} by: {self.max_prob*100:.2f} %"
            self.alarmPlayer.play()
            self.alarmPlaying = True
        return self.recognition_label






