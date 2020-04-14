import os
import cv2

# Directory Settings
CASCADES_DIR = 'Cascades/'
DATASET = 'Dataset/'

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Cascade Settings
CASCADE = cv2.CascadeClassifier(CASCADES_DIR + 'frontal.xml')


NAMES = ['None', 'Navaz', 'Quadri', 'Sanchit']