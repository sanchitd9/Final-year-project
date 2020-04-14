from face_recognition.camera_handle import CameraHandle
import numpy as np
import cv2

from face_recognition.face_detector import FaceDetector

from face_recognition.face_recognizer import FaceRecognizer
# cam = CameraHandle()

# while True:
#     ret, frame = cam.getFrame()
#     # frame = cv2.flip(frame, -1)
#     grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame', frame)
#     cv2.imshow('gray', grayscale_image)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# del cam


fr = FaceRecognizer()

print(fr.startDetection())