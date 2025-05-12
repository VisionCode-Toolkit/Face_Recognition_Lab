import cv2
import joblib
import numpy as np
import os
import random
class Face_recognizer():
    def __init__(self, output_recognition_viewer, output_detection_viewer):
        self.output_recognition_viewer = output_recognition_viewer
        self.output_detection_viewer = output_detection_viewer
        self.pca_features = None
        self.model = joblib.load("D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\classes\svm_pca_model.pkl")
        self.threshold = 0.2
        self.output_label = None
        self.train_data_dir = "D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\data\ORL database"
        self.outside_img = cv2.imread("D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\data\staff_only.jpg")
    def apply_face_recognition(self):
        # print(self.output_detection_viewer)
        self.pca_features = self.output_detection_viewer.current_image.pca_features
        print(self.pca_features.shape)
        probs = self.model.predict_proba(self.pca_features.reshape(1, -1))
        print(probs)
        max_prob = np.max(probs)
        if max_prob >= self.threshold:
            predicted_label = self.model.predict(self.pca_features.reshape(1, -1))[0]
            self.output_label = predicted_label
            print(f"The output label is {self.output_label}")
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
        else :
            print("outtttt")
            img_bgr = cv2.cvtColor(self.outside_img, cv2.COLOR_RGB2BGR)
            self.output_recognition_viewer.setImage(cv2.transpose(img_bgr))






