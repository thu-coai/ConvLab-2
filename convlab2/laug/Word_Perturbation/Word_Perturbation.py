#coding: UTF-8
from convlab2.laug.Word_Perturbation.multiwoz.multiwoz_eda import MultiwozEDA
from convlab2.laug.Word_Perturbation.multiwoz.aug_with_sgd_db import multiwoz_eda_config
from convlab2.laug.Word_Perturbation.frames.aug_with_sgd_db import frames_eda_config
class Word_Perturbation:
    def __init__(self,dataset='multiwoz'):
        self.dataset=dataset
        if dataset=='multiwoz':
            multiwoz_config=multiwoz_eda_config()
            self.EDA=MultiwozEDA(multiwoz_config.multiwoz,multiwoz_config.db_loader)
        elif dataset=='frames':
            frames_config=frames_eda_config()
            self.EDA=MultiwozEDA(frames_config.frames,frames_config.db_loader)
    def aug(self,text,span_info):
        (new_text,new_span_info,_),_=self.EDA.augment_sentence_only(text, span_info, {})
        return new_text,new_span_info

if __name__=="__main__":
    text = "I want a train to Cambridge"
    span_info = [["Train-Infrom","Dest","Cambridge",5,5]]
    WP = Word_Perturbation()
    new_text,new_span_info = WP.aug(text,span_info)
    print(new_text)
    print(new_span_info)
