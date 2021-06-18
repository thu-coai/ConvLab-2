#coding: UTF-8
from gtts import gTTS
from pydub.audio_segment import AudioSegment
import os


def text2wav(text,language='en',filename='temp',tld='cn'):
    gTTS(text=text, tld=tld,lang=language).save(filename+".mp3")
    AudioSegment.from_mp3(filename+".mp3").set_frame_rate(16000).export(filename+".wav", format="wav")

