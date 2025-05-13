import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QVBoxLayout, QSlider, QComboBox, QPushButton, \
    QStackedWidget, QWidget, QFileDialog, QRadioButton, QDialog, QLineEdit, QHBoxLayout, QSpinBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
from classes.image import Image
from classes.image_viewer import ImageViewer
from enums.viewerType import ViewerType
from classes.controller import Controller
from classes.Roc import Roc
import cv2

from classes.PCA import PCA

from classes.face_detector import Face_detector
from classes.face_recognizer import Face_recognizer

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)

        self.pca_features = None
        
        self.input_viewer_layout = self.findChild(QVBoxLayout,'input_layout')
        self.input_viewer = ImageViewer()
        self.input_viewer.viewer_type = ViewerType.INPUT
        self.input_viewer_layout.addWidget(self.input_viewer)
        
        self.detection_viewer_layout = self.findChild(QVBoxLayout,'detection_layout')
        self.detection_viewer = ImageViewer()
        self.detection_viewer.viewer_type = ViewerType.DETECTION
        self.detection_viewer_layout.addWidget(self.detection_viewer)


        self.recognition_viewer_layout = self.findChild(QVBoxLayout,'recognition_layout')
        self.recognition_viewer = ImageViewer()
        self.recognition_viewer.viewer_type = ViewerType.RECOGNITION
        self.recognition_viewer_layout.addWidget(self.recognition_viewer)
        
        self.controller = Controller(self.input_viewer, self.detection_viewer, self.recognition_viewer)
        
        self.browse_button = self.findChild(QPushButton, "browse")
        self.browse_button.clicked.connect(self.browse_)
        
        self.reset_button = self.findChild(QPushButton, "pushButton_2")
        self.reset_button.clicked.connect(self.reset)

        # detection stuff
        self.face_detector = Face_detector(self.detection_viewer)
        self.face_detector_button = self.findChild(QPushButton, "detection_output")
        self.face_detector_button.clicked.connect(self.apply_face_detector)

       # recognition
        self.recognition_label = self.findChild (QLabel,'match_name')
        self.face_recognizer = Face_recognizer(self.recognition_viewer, self.detection_viewer)
        self.recognition_button = self.findChild(QPushButton, "recognize_button")
        self.recognition_button.clicked.connect(self.apply_face_recognition)


        self.pca = PCA(image_viewer = self.detection_viewer)

    def apply_face_recognition(self):
        self.pca_features = self.pca.transform()
        self.detection_viewer.current_image.pca_features = self.pca_features
        self.recognition_label.setText(self.face_recognizer.apply_face_recognition() )
        # print(self.pca_features.shape)


    def reset(self):
        self.detection_viewer.current_image.reset()
        self.detection_viewer.clear()
        self.recognition_viewer.clear()
        self.controller.update()
        
    def make_roc_curve(self):
        Roc().make_roc(classes_per_plot=3, save = True)
        
    def browse_(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.JPG);;All Files (*)')
        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.JPG'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                self.input_viewer.current_image = image
                self.detection_viewer.current_image = image
                height, width, z = self.detection_viewer.current_image.modified_image.shape
                self.controller.update()

    def apply_face_detector(self):
        self.face_detector.apply_face_detection()
        self.controller.update()
                
                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())