"""
Dataloader base class. Every dataset should inherit this class and implement its own dataloader.
"""
from abc import ABC, abstractmethod
import os
import json
import sys
import zipfile
from pprint import pprint
from convlab2.util.file_util import read_zipped_json


class DatasetDataloader(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        load data from file, according to what is need
        :param args:
        :param kwargs:
        :return: data
        """
        pass

from convlab2 import DATA_ROOT

class MultiWOZDataloader(DatasetDataloader):
    def __init__(self, zh=False):
        super(MultiWOZDataloader, self).__init__()
        self.zh = zh

    def load_data(self,
                  data_dir=None,
                  data_key='all',
                  role='all',
                  utterance=False,
                  dialog_act=False,
                  context=False,
                  context_window_size=0,
                  context_dialog_act=False,
                  belief_state=False,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  ontology=False,
                  session_id=False,
                  span_info=False,
                  terminated=False,
                  goal=False
                  ):
        if data_dir is None:
            data_dir = os.path.join(DATA_ROOT, 'multiwoz' + ('_zh' if self.zh else ''))

        def da2tuples(dialog_act):
            tuples = []
            for domain_intent, svs in dialog_act.items():
                for slot, value in sorted(svs, key=lambda x: x[0]):
                    domain, intent = domain_intent.split('-')
                    tuples.append([intent, domain, slot, value])
            return tuples

        assert role in ['sys', 'usr', 'all']
        info_list = list(filter(eval, ['utterance', 'dialog_act', 'context', 'context_dialog_act', 'belief_state',
                                       'last_opponent_utterance', 'last_self_utterance', 'session_id', 'span_info',
                                       'terminated', 'goal']))
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role, 'human_val': {}}
        if data_key == 'all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            data = read_zipped_json(os.path.join(data_dir, '{}.json.zip'.format(data_key)), '{}.json'.format(data_key))
            print('loaded {}, size {}'.format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for sess_id, sess in data.items():
                cur_context = []
                cur_context_dialog_act = []
                for i, turn in enumerate(sess['log']):
                    text = turn['text']
                    da = da2tuples(turn['dialog_act'])
                    if role == 'sys' and i % 2 == 0:
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)
                        continue
                    elif role == 'usr' and i % 2 == 1:
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)
                        continue
                    if utterance:
                        self.data[data_key]['utterance'].append(text)
                    if dialog_act:
                        self.data[data_key]['dialog_act'].append(da)
                    if context:
                        self.data[data_key]['context'].append(cur_context[-context_window_size:])
                    if context_dialog_act:
                        self.data[data_key]['context_dialog_act'].append(cur_context_dialog_act[-context_window_size:])
                    if belief_state:
                        self.data[data_key]['belief_state'].append(turn['metadata'])
                    if last_opponent_utterance:
                        self.data[data_key]['last_opponent_utterance'].append(
                            cur_context[-1] if len(cur_context) >= 1 else '')
                    if last_self_utterance:
                        self.data[data_key]['last_self_utterance'].append(
                            cur_context[-2] if len(cur_context) >= 2 else '')
                    if session_id:
                        self.data[data_key]['session_id'].append(sess_id)
                    if span_info:
                        self.data[data_key]['span_info'].append(turn['span_info'])
                    if terminated:
                        self.data[data_key]['terminated'].append(i + 2 >= len(sess['log']))
                    if goal:
                        self.data[data_key]['goal'].append(sess['goal'])
                    cur_context.append(text)
                    cur_context_dialog_act.append(da)
        if ontology:
            ontology_path = os.path.join(data_dir, 'ontology.json')
            self.data['ontology'] = json.load(open(ontology_path))

        return self.data


class CamrestDataloader(DatasetDataloader):
    def __init__(self):
        super(CamrestDataloader, self).__init__()

    def load_data(self,
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/camrest')),
                  data_key='all',
                  role='all',
                  utterance=False,
                  dialog_act=False,
                  context=False,
                  context_window_size=0,
                  context_dialog_act=False,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  session_id=False,
                  terminated=False,
                  goal=False
                  ):

        def da2tuples(dialog_act):
            tuples = []
            for intent, svs in dialog_act.items():
                for slot, value in sorted(svs, key=lambda x: x[0]):
                    tuples.append([intent, slot, value])
            return tuples

        assert role in ['sys', 'usr', 'all']
        info_list = list(filter(eval, ['utterance', 'dialog_act', 'context', 'context_dialog_act',
                                       'last_opponent_utterance', 'last_self_utterance', 'session_id',
                                       'terminated', 'goal']))
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role}
        if data_key == 'all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            data = read_zipped_json(os.path.join(data_dir, '{}.json.zip'.format(data_key)), '{}.json'.format(data_key))
            print('loaded {}, size {}'.format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for sess in data:
                cur_context = []
                cur_context_dialog_act = []
                for turn in sess['dial']:
                    turn_id = turn['turn']
                    for side_id in ['usr', 'sys']:
                        if side_id == 'usr':
                            text = turn[side_id]['transcript']
                        else:
                            text = turn[side_id]['sent']
                        da = da2tuples(turn[side_id]['dialog_act'])
                        if {role, side_id} == {'usr', 'sys'}:
                            cur_context.append(text)
                            cur_context_dialog_act.append(da)
                            continue
                        if utterance:
                            self.data[data_key]['utterance'].append(text)
                        if dialog_act:
                            self.data[data_key]['dialog_act'].append(da)
                        if context and context_window_size:
                            self.data[data_key]['context'].append(cur_context[-context_window_size:])
                        if context_dialog_act and context_window_size:
                            self.data[data_key]['context_dialog_act'].append(
                                cur_context_dialog_act[-context_window_size:])
                        if last_opponent_utterance:
                            self.data[data_key]['last_opponent_utterance'].append(
                                cur_context[-1] if len(cur_context) >= 1 else '')
                        if last_self_utterance:
                            self.data[data_key]['last_self_utterance'].append(
                                cur_context[-2] if len(cur_context) >= 2 else '')
                        if session_id:
                            self.data[data_key]['session_id'].append(sess['dialogue_id'])
                        if terminated:
                            self.data[data_key]['terminated'].append(turn_id >= len(sess['dial']))
                        if goal:
                            self.data[data_key]['goal'].append(sess['goal'])
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)

        return self.data


class CrossWOZDataloader(DatasetDataloader):
    def __init__(self, en=False):
        super(CrossWOZDataloader, self).__init__()
        self.en = en

    def load_data(self,
                  data_dir=None,
                  data_key='all',
                  role='all',
                  utterance=False,
                  dialog_act=False,
                  context=False,
                  context_window_size=0,
                  context_dialog_act=False,
                  user_state=False,
                  sys_state=False,
                  sys_state_init=False,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  session_id=False,
                  terminated=False,
                  goal=False,
                  final_goal=False,
                  task_description=False
                  ):
        if data_dir is None:
            data_dir = os.path.join(DATA_ROOT, 'crosswoz' + ('_en' if self.en else ''))

        def da2tuples(dialog_act):
            tuples = []
            for act in dialog_act:
                tuples.append([act[0], act[1], act[2], act[3]])
            return tuples

        assert role in ['sys', 'usr', 'all']
        info_list = list(filter(eval, ['utterance', 'dialog_act', 'context', 'context_dialog_act',
                                       'user_state', 'sys_state', 'sys_state_init',
                                       'last_opponent_utterance', 'last_self_utterance', 'session_id',
                                       'terminated', 'goal', 'final_goal', 'task_description']))
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role, 'human_val': {}}
        if data_key == 'all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            data = read_zipped_json(os.path.join(data_dir, '{}.json.zip'.format(data_key)), '{}.json'.format(data_key))
            print('loaded {}, size {}'.format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for sess_id, sess in data.items():
                cur_context = []
                cur_context_dialog_act = []
                for i, turn in enumerate(sess['messages']):
                    text = turn['content']
                    da = da2tuples(turn['dialog_act'])
                    if {role, turn['role']} == {'usr', 'sys'}:
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)
                        continue
                    if utterance:
                        self.data[data_key]['utterance'].append(text)
                    if dialog_act:
                        self.data[data_key]['dialog_act'].append(da)
                    if context and context_window_size:
                        self.data[data_key]['context'].append(cur_context[-context_window_size:])
                    if context_dialog_act and context_window_size:
                        self.data[data_key]['context_dialog_act'].append(cur_context_dialog_act[-context_window_size:])
                    if role in ['usr', 'all'] and user_state and turn['role'] == 'usr':
                        self.data[data_key]['user_state'].append(turn['user_state'])
                    if role in ['sys', 'all'] and sys_state and turn['role'] == 'sys':
                        self.data[data_key]['sys_state'].append(turn['sys_state'])
                    if role in ['sys', 'all'] and sys_state_init:
                        if turn['role'] == 'sys':
                            self.data[data_key]['sys_state_init'].append(turn['sys_state_init'])
                        else:
                            self.data[data_key]['sys_state_init'].append({})
                    if last_opponent_utterance:
                        self.data[data_key]['last_opponent_utterance'].append(
                            cur_context[-1] if len(cur_context) >= 1 else '')
                    if last_self_utterance:
                        self.data[data_key]['last_self_utterance'].append(
                            cur_context[-2] if len(cur_context) >= 2 else '')
                    if session_id:
                        self.data[data_key]['session_id'].append(sess_id)
                    if terminated:
                        self.data[data_key]['terminated'].append(i + 2 >= len(sess['messages']))
                    if goal:
                        self.data[data_key]['goal'].append(sess['goal'])
                    if final_goal:
                        self.data[data_key]['final_goal'].append(sess['final_goal'])
                    if task_description:
                        self.data[data_key]['task_description'].append(sess['task description'])
                    cur_context.append(text)
                    cur_context_dialog_act.append(da)

        return self.data


class DealOrNotDataloader(DatasetDataloader):
    def __init__(self):
        super(DealOrNotDataloader, self).__init__()

    def load_data(self,
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/deal_or_not')),
                  data_key='all',
                  role='all',
                  utterance=False,
                  context=False,
                  context_window_size=0,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  session_id=False,
                  terminated=False,
                  goal=False,
                  output=False,
                  partner_input=False
                  ):

        def get_tag(tokens, tag):
            return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]

        assert role in ['YOU', 'THEM', 'all']
        info_list = list(
            filter(eval, ['utterance', 'context', 'last_opponent_utterance', 'last_self_utterance', 'session_id',
                          'terminated', 'goal', 'output', 'partner_input']))
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role}
        if data_key == 'all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]

        for data_key in data_key_list:
            archive = zipfile.ZipFile(os.path.join(data_dir, 'deal_or_not.zip'), 'r')
            data = archive.open(f'{data_key}.txt', 'r').readlines()
            print('loaded {}, size {}'.format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for line in data:
                line = line.decode(encoding='utf-8')
                cur_context = []
                line = line.strip().split()
                dialog = get_tag(line, 'dialogue')
                first_role = dialog[0].strip(':')
                second_role = ['THEM', 'YOU'][first_role == 'THEM']
                count = 0
                while '<eos>' in dialog or '<selection>' in dialog:
                    count += 1
                    if '<eos>' in dialog:
                        text = ' '.join(dialog[1: dialog.index('<eos>')])
                        dialog = dialog[dialog.index('<eos>') + 1:]
                    elif '<selection>' in dialog:
                        text = '<selection>'
                        dialog = []
                    if role == first_role and count % 2 == 0:
                        cur_context.append(text)
                        continue
                    elif role == second_role and count % 2 == 1:
                        cur_context.append(text)
                        continue
                    if utterance:
                        self.data[data_key]['utterance'].append(text)
                    if context and context_window_size:
                        self.data[data_key]['context'].append(cur_context[-context_window_size:])
                    if last_opponent_utterance:
                        self.data[data_key]['last_opponent_utterance'].append(
                            cur_context[-1] if len(cur_context) >= 1 else '')
                    if last_self_utterance:
                        self.data[data_key]['last_self_utterance'].append(
                            cur_context[-2] if len(cur_context) >= 2 else '')
                    if session_id:
                        self.data[data_key]['session_id'].append(count)
                    if terminated:
                        self.data[data_key]['terminated'].append('<eos>' not in dialog)
                    if goal:
                        self.data[data_key]['goal'].append(get_tag(line, 'input'))
                    if output:
                        self.data[data_key]['output'].append(get_tag(line, 'output'))
                    if partner_input:
                        self.data[data_key]['partner_input'].append(get_tag(line, 'partner_input'))
                    cur_context.append(text)

        return self.data


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1]

        if dataset_name == 'MultiWOZ':
            m = MultiWOZDataloader()
            pprint(m.load_data(
                data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/multiwoz')),
                data_key='all',
                role='sys',
                utterance=True,
                dialog_act=True,
                context=True,
                context_window_size=0,
                context_dialog_act=True,
                belief_state=True,
                last_opponent_utterance=True,
                last_self_utterance=True,
                ontology=True,
                session_id=True,
                span_info=True,
                terminated=True,
                goal=True
            ))
        elif dataset_name == 'Camrest':
            m = CamrestDataloader()
            pprint(m.load_data(
                data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/camrest')),
                data_key='all',
                role='sys',
                utterance=True,
                dialog_act=True,
                context=True,
                context_window_size=0,
                context_dialog_act=True,
                last_opponent_utterance=True,
                last_self_utterance=True,
                session_id=True,
                terminated=True,
                goal=True
                ))

        elif dataset_name == 'CrossWOZ':
            m = CrossWOZDataloader()
            pprint(m.load_data(
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/crosswoz')),
                  data_key='all',
                  role='all',
                  utterance=True,
                  dialog_act=True,
                  context=True,
                  context_window_size=0,
                  context_dialog_act=True,
                  user_state=True,
                  sys_state=True,
                  sys_state_init=True,
                  last_opponent_utterance=True,
                  last_self_utterance=True,
                  session_id=True,
                  terminated=True,
                  goal=True,
                  final_goal=True,
                  task_description=True
                  ))

        elif dataset_name == 'DealOrNot':
            m = DealOrNotDataloader()
            pprint(m.load_data(
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/deal_or_not')),
                  data_key='all',
                  role='THEM',
                  utterance=True,
                  context=True,
                  context_window_size=0,
                  last_opponent_utterance=True,
                  last_self_utterance=True,
                  session_id=True,
                  terminated=True,
                  goal=True,
                  output=True,
                  partner_input=True
                  ))

        else:
            raise Exception("currently supported dataset: MultiWOZ, CrossWOZ, Camrest, DealOrNot")

    else:
        print("dataloader test usage:")
        print("\t python dataset_dataloader.py dataset")
        print("\t dataset=MultiWOZ, CrossWOZ, Camrest, or DealOrNot")
        sys.exit()
