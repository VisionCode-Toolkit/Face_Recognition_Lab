import cv2

class Face_detector():

    def __init__(self, output_detection_viewer):
        self.output_detection_viewer = output_detection_viewer
        # loading haar cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def apply_face_detection(self):
        original_image = self.output_detection_viewer.current_image.modified_image
        # make sure its gray --> haar cascade classifier only works with gray imgs
        if len(self.output_detection_viewer.current_image.modified_image.shape) == 3:
            print("img is not gray")
            self.output_detection_viewer.current_image.transfer_to_gray_scale()

        # 1.05 is value for using a small step for resizing (reduce size by 5%) --> lower leads to better acc
        # 5 is min neighbors (will affect the quality of the detected faces)
        detected_faces = self.face_cascade.detectMultiScale(self.output_detection_viewer.current_image.modified_image, 1.3, 5)

        self.output_detection_viewer.current_image.modified_image = self.draw_rects_on_faces(original_image,detected_faces )



    def draw_rects_on_faces(self, original_image, detected_faces):
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return original_image
