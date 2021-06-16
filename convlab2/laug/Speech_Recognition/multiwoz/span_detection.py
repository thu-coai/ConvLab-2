from .detection_utils import translateNumberToEnglish,translateTimeToEnglish,span_typer,replacer
import json
from .paraphrase_span_detection import phrase_idx_utt

def span_detect(original_text,new_text,span_list):
#input:original_text,new_text,one span_info [slot,slot,span,start,end]
#output:is_span_found? , is_span_changed? , new span_info [slot,slot,new span,new start,new end]
    span=span_list[2].lower()
    span=replacer(span)
    span_type=span_typer(span)
    new_words=new_text.split()
    if span_type=="time":
        span2=translateTimeToEnglish(span)
    if span_type=="number":
        span2=translateNumberToEnglish(span)
    if span_type=="none":
        span2=span
    span_changed,span_found=0,0
    if new_text.find(span)>=0:
        span_changed,span_found=0,1
        span_start=new_text.count(' ',0,new_text.find(span))
        span_end=span_start+len(span.split())-1
        new_span_list=[span_list[0],span_list[1],' '.join(new_words[span_start:span_end+1]),span_start,span_end]
    elif new_text.find(span2)>=0:
        span_changed,span_found=1,1
        span=span2
        span_start=new_text.count(' ',0,new_text.find(span))
        span_end=span_start+len(span.split())-1
        new_span_list=[span_list[0],span_list[1],' '.join(new_words[span_start:span_end+1]),span_start,span_end]
    else:
        span=span2
        span_words=span.split()
        
        result=phrase_idx_utt(span_words,new_words)
        if result is not None:
            max_start,max_end=result
            span_changed,span_found=1,1
            new_span_list=[span_list[0],span_list[1],' '.join(new_words[max_start:max_end+1]),max_start,max_end]
        else:
            origin_split=original_text.split() 
            new_split=new_words
            ok=0
            origin_start=span_list[3]-1
            if    origin_start>=0:
                if origin_start-1>=0 and origin_split[origin_start] in ['.',',','?']:
                    origin_start-=1
                start_word=origin_split[origin_start]
                for start in range(len(new_split)):
                    if new_split[start]==start_word:
                        break
                start+=1
            else:
                start=0
            if span_list[4]+1<len(origin_split) and start<len(new_split):
                end_word=origin_split[span_list[4]+1]
                if end_word not in ['.',',','?']:
                    if span_list[4]+1<len(origin_split):
                        end_word=origin_split[span_list[4]+1]
                        for end in range(start,len(new_split)):
                            if new_split[end]==end_word:
                                ok=1
                                break
                        end-=1
                     
                else:
                    if span_list[4]+2<len(origin_split):
                        end_word=origin_split[span_list[4]+2]
                        for end in range(start,len(new_split)):
                            if new_split[end]==end_word:
                                ok=1
                                break
                        end-=1
                    else:
                        ok=1
                        end=len(new_split)-1    
            else:
                ok=1
                end=len(new_split)-1
            if start<=end and ok==1:
                span_changed,span_found=1,1
                new_span_list=[span_list[0],span_list[1],' '.join(new_words[start:end+1]),start,end]
        
    if span_found==0:
        new_span_list=[span_list[0],span_list[1],span_list[2],0,0]
    return new_span_list
