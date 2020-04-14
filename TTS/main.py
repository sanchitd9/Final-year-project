from gtts import gTTS
import os

class TTS:
    def __init__(self, text):
        self.text = text
        self.language = 'en'
        self.obj = gTTS(text = self.text, lang = self.language, slow = False)
        self.obj.save('TTS/audio.mp3')


    def play(self):
        os.system("mpg321 TTS/audio.mp3")



