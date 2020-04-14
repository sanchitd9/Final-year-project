import numpy as np
import cv2
from .camera_handle import CameraHandle
from Config.config import CASCADE
import os

class FaceDetector:
    def __init__(self):
        self.faceCascade = CASCADE