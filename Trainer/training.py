import cv2
import numpy as np
from PIL import Image
import os
from Config.config import DATASET
from face_recognition.face_detector import FaceDetector


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = FaceDetector()

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L')
        image_matrix = np.array(PIL_image, 'uint8')
        id = int(os.path.split(imagePath)[-1].split('-')[1])
        faces = detector.faceCascade.detectMultiScale(image_matrix)

        for (x, y, w, h) in faces:
            faceSamples.append(image_matrix[y:y+h, x:x+w])
            ids.append(id)
        
    return faceSamples, ids

print('Training model...')

faces, ids = getImagesAndLabels(DATASET)

recognizer.train(faces, np.array(ids))

recognizer.write('Trainer/trainer.yml')

print('Done!')