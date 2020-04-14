import cv2
import numpy as np
import os
from .camera_handle import CameraHandle
from .face_detector import FaceDetector
from Config.config import NAMES


class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('Trainer/trainer.yml')
        self.detector = FaceDetector()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.id = 0
        self.cam = CameraHandle()
        self.minW, self.minH = self.cam.setWandH(3, 4)


    def startDetection(self):

        self.name = 'Unknown'
        self.flag = 0


        while True:
            self.ret, self.image = self.cam.camera.read()
            self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            self.faces = self.detector.faceCascade.detectMultiScale(self.grayscale_image, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(self.minW), int(self.minH)))

            for (x, y, w, h) in self.faces:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.id, self.confidence = self.recognizer.predict(self.grayscale_image[y:y+h, x:x+w])

                if self.confidence < 100:
                    self.id = NAMES[self.id]
                    self.name = self.id
                    self.confidence = ' {0}%'.format(round(100 - self.confidence))
                    self.flag = 1
                    break
                else:
                    self.id = 'unknown'
                    self.confidence = ' {0}%'.format(round(100 - self.confidence))
                    self.flag = 1
                    break
        
                    # cv2.putText(self.image, str(self.id), (x+5, y-5), self.font, 1, (255, 255, 255), 2)
                    # cv2.putText(self.image, str(self.confidence), (x+5, y+h-5), self.font, 1, (255, 255, 0), 1)

                    
            if self.flag == 1:
                break
                

            # cv2.imshow('camera', self.image)

            # self.k = cv2.waitKey(10) & 0xff
            # if self.k == 27:
            #     break

        del self.cam
        return self.name