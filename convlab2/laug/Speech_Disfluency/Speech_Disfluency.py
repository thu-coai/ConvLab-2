# -*- coding: utf-8 -*-
import json
import random
from fuzzywuzzy import fuzz
from convlab2.laug.Speech_Disfluency.inference import IP_model
import os

current_path=os.path.dirname(os.path.abspath(__file__))
def random_01(possibility):
    x=random.random()
    if x>=possibility:
        return 0
    else:
        return 1
        
def random_pick_from_list(random_list):
    return random_list[int(len(random_list)*random.random())]

def process_distribution_dict(distribution_dict):
    processed_distribution=[]
    sum=0
    for key in distribution_dict:
        sum+=distribution_dict[key]
        processed_distribution.append((key,sum))
    return processed_distribution

def random_pick_from_distribution(distribution_dict):
    processed_distribution=process_distribution_dict(distribution_dict)
    x=random.random()*processed_distribution[-1][1]
    for item in processed_distribution:
        if x>item[1]:
            continue
        else:
            picked_item=item[0]
            break
    return picked_item

def preprocess(sentence):
    word_list=sentence.lower().strip().split()
    return word_list    
    
class Speech_Disfluency:
    def __init__(self,dataset='multiwoz',edit_frequency=0.3):
        self.resources=json.load(open(os.path.join(current_path,'resources/resources_'+dataset+'.json'),'r'))
        self.edit_frequency=edit_frequency
        

    def protect_slots(self,word_list,spans,IP_tags):
        sentence=' '.join(word_list)+' '
        for span in spans:
            value=span[2]
            start=sentence.count(' ',0,sentence.find(' '+value+' '))
            lenth=len(value.split())
            for i in range(start+1,start+lenth):
                IP_tags[i]=0
                IP_tags[start]=1
            if IP_tags[start]==2:
                IP_tags[start]=1
        return IP_tags
    
    
    def add_repairs(self,word_list,spans):
        sentence=' '+' '.join(word_list)+' '
        if len(spans)==0:
            return word_list
        else:
            edit_possibility=self.edit_frequency/len(spans)
        for span in spans:
            if random_01(edit_possibility)==0:
                continue
            value=span[2]
            start=sentence.count(' ',0,sentence.find(' '+value+' '))-1
            
            max_ratio,max_entity=0,''
            for e in self.resources["knowledge_base"]["entity"]:
                ratio=fuzz.ratio(e,value)
                if ratio>max_ratio:
                    max_ratio=ratio
                    max_entity=e
            if max_entity!='' and max_ratio>60:
                candidate=[]
                if max_entity in self.resources["knowledge_base"]["entity"]:
                    candidate=self.resources["knowledge_base"]["category"][random_pick_from_list(self.resources["knowledge_base"]["entity"][max_entity])][0:]
                if span in candidate:
                    candidate.remove(span)
                if len(candidate)!=0:
                    word_list[start]=random_pick_from_list(candidate)+' '+random_pick_from_list(self.resources["edit_terms"])+' '+word_list[start]
        return word_list
        
    def add_repeats(self,word_list,IP_tags):
        for i in range(len(IP_tags)):
            if IP_tags[i]==2:
                word_list[i]=word_list[i]+random_pick_from_list([' ',' , '])+word_list[i]
        return word_list
    
        
    def add_fillers(self,word_list,IP_tags):
        for i in range(len(IP_tags)):
            if IP_tags[i]==1:
                word_list[i]=random_pick_from_distribution(self.resources["filler_terms"])+' '+word_list[i]
        return word_list
        
    def add_restart(self,word_list):
        word_list[0]=random_pick_from_distribution(self.resources["restart_terms"])+' '+word_list[0]
        return word_list
    
        
    def find_spans(self,disfluent_sentence,spans):
        checked=1
        sentence=' '+disfluent_sentence+' '
        for i in range(len(spans)):
            value=spans[i][2]
            start=sentence.count(' ',0,sentence.find(' '+value+' '))
            lenth=len(value.split())
            spans[i][3]=start
            spans[i][4]=start+lenth-1
            if ' '.join(sentence.split()[spans[i][3]:spans[i][4]+1])!=spans[i][2]:
                checked=0
        return spans,checked
        
    def aug(self,sentence,spans):
        word_list=preprocess(sentence)
        IP_tags=IP_model(word_list)
        IP_tags=self.protect_slots(word_list,spans,IP_tags)
        word_list=self.add_repairs(word_list,spans)
        word_list=self.add_repeats(word_list,IP_tags)
        word_list=self.add_fillers(word_list,IP_tags)
        word_list=self.add_restart(word_list)
        disfluent_sentence=' '.join(word_list)
        new_spans,checked=self.find_spans(disfluent_sentence,spans)
        return disfluent_sentence,new_spans
    # input sentence and span_info ; output the disfluent sentence and new_span_info
    
if __name__=="__main__":
    text = "I want a train to Cambridge"
    span_info = [["Train-Inform","Dest","Cambridge",5,5]]
    SR = Speech_Disfluency()
    new_text,new_span_info = SR.aug(text,span_info)
    print(new_text)
    print(new_span_info)
