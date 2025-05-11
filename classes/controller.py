class Controller():
    def __init__(self, input_viewer, detection_viewer, recognition_viewer):
        self.input_viewer = input_viewer
        self.detection_viewer = detection_viewer
        self.recognition_viewer = recognition_viewer
        
        
    def update(self):
        self.input_viewer.update_plot()
        self.detection_viewer.update_plot()
        self.recognition_viewer.update_plot()