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
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        
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
    
    def reset(self):
        self.output_viewer.current_image.reset()
        self.controller.update()
        
        
    def browse_(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.jpeg *.jpg *.png *.JPG);;All Files (*)')
        if file_path:
            if file_path.endswith('.jpeg') or file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.JPG'):
                temp_image = cv2.imread(file_path)
                image = Image(temp_image)
                self.input_viewer.current_image = image
                self.output_viewer.current_image = image
                height, width, z = self.output_viewer.current_image.modified_image.shape
                self.controller.update()
                
                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())