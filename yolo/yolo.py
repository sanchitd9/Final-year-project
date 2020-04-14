import cv2
import numpy as np
import os
from picamera import PiCamera
from Config.config import CLASS_NAMES


class ObjectDetector:

    def __init__(self):
        self.model = cv2.dnn.readNetFromTensorflow('yolo/frozen_inference_graph.pb', 'yolo/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.answer = []

    def id_class_name(self, class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value

    def detectObject(self):
        self.cam = PiCamera()
        self.cam.start_preview()
        self.cam.capture('yolo/img.jpg')
        self.cam.close()
        
        self.image = cv2.imread('yolo/img.jpg')
        self.model.setInput(cv2.dnn.blobFromImage(self.image, size = (640, 480), swapRB = True))
        self.output = self.model.forward()
        self.image_height, self.image_width, _ = self.image.shape

        for detection in self.output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .5:
                class_id = detection[1]
                class_name = self.id_class_name(class_id, CLASS_NAMES)
                self.answer.append(class_name)
                # print(str(str(class_id) + " " + str(detection[2]) + " " + id_class_name(class_id, CLASS_NAMES)))
                box_x = detection[3] * self.image_width
                box_y = detection[4] * self.image_height
                box_width = detection[5] * self.image_width
                box_height = detection[6] * self.image_height
                # cv2.rectangle(self.image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                # cv2.putText(self.image, class_name ,(int(box_x), int(box_y+.05*self.image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*self.image_width),(0, 0, 255))
        return self.answer


    def __del__(self):
        cv2.destroyAllWindows()

    
