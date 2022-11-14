# -*- coding: utf-8 -*-
from convlab2.nlg.scgpt.multiwoz.scgpt import SCGPT
from convlab2.laug.Text_Paraphrasing.utils import span2tuple,paraphrase_span_detection
class Text_Paraphrasing:
    def __init__(self,dataset='multiwoz'):
        if dataset=='multiwoz':
            self.model=SCGPT()
        if dataset=='frames':
            self.model=SCGPT(model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/nlg-gpt-frames.zip')
        self.model.init_session()
    def aug(self,text,span_info):
        t=span2tuple(span_info)
        new_text = self.model.generate(t)
        new_span_info = paraphrase_span_detection(new_text,span_info)
        return new_text, new_span_info

        
if __name__=="__main__":
    text = "I want a train to Cambridge"
    span_info = [["Train-Infrom","Dest","Cambridge",5,5]]
    TP = Text_Paraphrasing()
    new_text,new_span_info = TP.aug(text,span_info)
    print(new_text)
    print(new_span_info)
