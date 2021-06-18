from convlab2.laug import Word_Perturbation
from convlab2.laug import Text_Paraphrasing
from convlab2.laug import Speech_Recognition
from convlab2.laug import Speech_Disfluency

if __name__=="__main__":
    text = "I want a train to Cambridge"
    span_info = [["Train-Infrom","Dest","Cambridge",5,5]]
    WP = Word_Perturbation('multiwoz')
    TP = Text_Paraphrasing('multiwoz')
    SR = Speech_Recognition('multiwoz')
    SD = Speech_Disfluency('multiwoz')
    WP_text,WP_span_info = WP.aug(text,span_info)
    print('Word Perturbation:')
    print(WP_text)
    print(WP_span_info)
    TP_text,TP_span_info = TP.aug(text,span_info)
    print('Text Paraphrasing:')
    print(TP_text)
    print(TP_span_info)
    SR_text,SR_span_info = SR.aug(text,span_info)
    print('Speech Recognition:')
    print(SR_text)
    print(SR_span_info)
    SD_text,SD_span_info = SD.aug(text,span_info)
    print('Speech Disfluency:')
    print(SD_text)
    print(SD_span_info)
    
