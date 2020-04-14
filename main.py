from TTS.main import TTS
from face_recognition.face_recognizer import FaceRecognizer
from yolo.yolo import ObjectDetector

def main():

    fr = FaceRecognizer()
    ot = ObjectDetector()

    while True:
        ch = int(input('1. Detect Face\n2. Detect Objects\nEnter your choice:\n'))
        if ch == 1:
            name = fr.startDetection()
            name_speech = TTS(str(name))
            name_speech.play()
        elif ch == 2:
            objects = ot.detectObject()
            output = ''
            for item in objects:
                output += str(item)
                output += ','
            output_speech = TTS(output)
            output_speech.play()
        else:
            break

if __name__ == '__main__':
    main()