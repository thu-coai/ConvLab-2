import os
import sys
from convlab2.policy.hdsa.multiwoz.predictor import HDSA_predictor
from convlab2.policy.hdsa.multiwoz.generator import HDSA_generator
from convlab2.policy import Policy

DEFAULT_DIRECTORY = "model"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "hdsa.zip")


class HDSA(Policy):

    def __init__(self, archive_file=DEFAULT_ARCHIVE_FILE, model_file="https://convlab.blob.core.windows.net/convlab-2/hdsa.zip", use_cuda=False):
        self.predictor = HDSA_predictor(archive_file, model_file, use_cuda)
        self.generator = HDSA_generator(archive_file, model_file, use_cuda)

    def init_session(self):
        self.generator.init_session()

    def predict(self, state):

        act, kb = self.predictor.predict(state)
        response = self.generator.generate(state, act, kb)

        return response


if __name__ == '__main__':

    state = {'user_action': [["Inform", "Hotel", "Area", "east"], ["Inform", "Hotel", "Stars", "4"]],
    'system_action': [],
    'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                    'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                'semi': {'name': '',
                                        'area': 'east',
                                        'parking': '',
                                        'pricerange': '',
                                        'stars': '4',
                                        'internet': '',
                                        'type': ''}},
                    'attraction': {'book': {'booked': []},
                                    'semi': {'type': '', 'name': '', 'area': ''}},
                    'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                    'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                    'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                    'taxi': {'book': {'booked': []},
                            'semi': {'leaveAt': '',
                                        'destination': '',
                                        'departure': '',
                                        'arriveBy': ''}},
                    'train': {'book': {'booked': [], 'people': ''},
                                'semi': {'leaveAt': '',
                                        'destination': '',
                                        'day': '',
                                        'arriveBy': '',
                                        'departure': ''}}},
    'request_state': {},
    'terminated': False,
    'history': [['sys', ''],
        ['user', 'Could you book a 4 stars hotel east of town for one night, 1 person?']]}

    cur_model = HDSA()
    response = cur_model.predict(state)
    print(response)

