import numpy as np
import cv2
from Config.config import CAMERA_WIDTH, CAMERA_HEIGHT


class CameraHandle:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, CAMERA_WIDTH)
        self.camera.set(4, CAMERA_HEIGHT)


    def getFrame(self):
        return self.camera.read()

    def setWandH(self, w, h):
        self.minW = 0.1*self.camera.get(w)
        self.minH = 0.1*self.camera.get(h)
        return (self.minW, self.minH)

    def __del__(self):
        self.camera.release()
        cv2.destroyAllWindows()