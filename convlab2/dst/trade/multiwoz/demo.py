import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from convlab2.dst.trade.multiwoz.utils.config import *
from convlab2.dst.trade.multiwoz.trade import *


'''
python demo.py
Note: To support Jupyter Notebook, we have disabled args in command line. You can specify
    your own parameters in parser in utils/config.py
'''

a = args
# specify model path
# args['path'] = 'model/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.3591'
# model = MultiWOZTRADE(args['path'])
model = MultiWOZTRADE()

user_act = 'i need to book a hotel in the east that has 4 stars .'
model.state['history'] = [['user', user_act]]
state = model.update(user_act)
print(state)