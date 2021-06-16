#coding: UTF-8
from convlab2.laug.Speech_Recognition.ASR import wav2text
from convlab2.laug.Speech_Recognition.TTS import text2wav
from convlab2.laug.Speech_Recognition.multiwoz.span_detection import span_detect

import os
import time

class Speech_Recognition:
    def __init__(self,dataset='multiwoz',temp_file='temp',tld='com'):
        
        self.wav2text = wav2text()
        self.temp_file = temp_file
        self.tld = tld
    def aug(self,text,span_info):
        ok=0
        while ok==0:
            try:
                text2wav(text,tld=self.tld,filename=self.temp_file)
            except ValueError:
                ok=0
                print("gTTS error occur!")
            else:
                ok=1
        new_text = self.wav2text.run(self.temp_file+".wav")
        new_span_info=[]
        for span in span_info:
            new_span_info.append(span_detect(text,new_text,span))
        return new_text,new_span_info

if __name__=="__main__":
    text = "I want a train to Cambridge"
    span_info = [["Train-Inform","Dest","Cambridge",5,5]]
    SR = Speech_Recognition()
    new_text,new_span_info = SR.aug(text,span_info)
    print(new_text)
    print(new_span_info)
