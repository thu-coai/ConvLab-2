# -*- coding: utf-8 -*-
from convlab2.util.multiwoz.paraphrase_span_detection import phrase_idx_utt

def paraphrase_span_detection(new_text,span_info):
    new_words=new_text.split()
    new_span_info=[]
    for span in span_info:
        span_words=span[2].split()
        result=phrase_idx_utt(span_words,new_words)
        if result is not None:
            max_start,max_end=result
            new_span_info.append([span[0],span[1],' '.join(new_words[max_start:max_end+1]),max_start,max_end])
    return new_span_info

    
def span2tuple(span_info):
    t=[]
    for span in span_info:
        t.append((span[0].split('-')[1],span[0].split('-')[0],span[1],span[2]))
    return t