#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__time__ = '2019/1/31 10:24'

import copy
import json
import os
import random
import re
import logging

from convlab2.policy.policy import Policy
from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_USR_DA, REF_SYS_DA

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]

# import reflect table
REF_USR_DA_M = copy.deepcopy(REF_USR_DA)
REF_SYS_DA_M = {}
for dom, ref_slots in REF_SYS_DA.items():
    dom = dom.lower()
    REF_SYS_DA_M[dom] = {}
    for slot_a, slot_b in ref_slots.items():
        if slot_a == 'Ref':
            slot_b = 'ref'
        REF_SYS_DA_M[dom][slot_a.lower()] = slot_b
    REF_SYS_DA_M[dom]['none'] = 'none'
REF_SYS_DA_M['taxi']['phone'] = 'phone'
REF_SYS_DA_M['taxi']['car'] = 'car type'

# def book slot
BOOK_SLOT = ['people', 'day', 'stay', 'time']


class UserPolicyAgendaMultiWoz(Policy):
    """ The rule-based user policy model by agenda. Derived from the UserPolicy class """

    # load stand value
    with open(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, 'data/multiwoz/value_set.json')) as f:
        stand_value_dict = json.load(f)

    def __init__(self):
        """
        Constructor for User_Policy_Agenda class.
        """
        self.max_turn = 40
        self.max_initiative = 4

        self.goal_generator = GoalGenerator()

        self.__turn = 0
        self.goal = None
        self.agenda = None

        Policy.__init__(self)

    def reset_turn(self):
        self.__turn = 0

    def init_session(self, ini_goal=None):
        """ Build new Goal and Agenda for next session """
        self.reset_turn()
        if not ini_goal:
            self.goal = Goal(self.goal_generator)
        else:
            self.goal = ini_goal
        self.domain_goals = self.goal.domain_goals
        self.agenda = Agenda(self.goal)

    def predict(self, sys_dialog_act):
        """
        Predict an user act based on state and preorder system action.
        Args:
            sys_dialog_act (list): system dialogue act: [[intent, domain, slot, value],...].
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by user.
        """
        self.__turn += 2

        assert isinstance(sys_dialog_act, list)

        sys_action = {}
        for intent, domain, slot, value in sys_dialog_act:
            if slot == 'Choice' and value.strip().lower() in ['0', 'zero']:
                nooffer_key = '-'.join([domain, 'NoOffer'])
                sys_action.setdefault(nooffer_key, [])
                sys_action[nooffer_key].append(['none', 'none'])
            else:
                k = '-'.join([domain, intent])
                sys_action.setdefault(k, [])
                sys_action[k].append([slot, value])

        if self.__turn > self.max_turn:
            self.agenda.close_session()
        else:
            sys_action = self._transform_sysact_in(sys_action)
            # print('sys action before update agenda', sys_action)
            self.agenda.update(sys_action, self.goal)
            if self.goal.task_complete():
                self.agenda.close_session()

        action = {}
        while len(action) == 0:
            # A -> A' + user_action
            # action = self.agenda.get_action(random.randint(2, self.max_initiative))
            action = self.agenda.get_action(self.max_initiative)

            # transform to DA
            action = self._transform_usract_out(action)
            # print(action)

        tuples = []
        for domain_intent, svs in action.items():
            for slot, value in svs:
                domain, intent = domain_intent.split('-')
                tuples.append([intent, domain, slot, value])

        return tuples

    def is_terminated(self):
        # Is there any action to say?
        return self.agenda.is_empty()

    def get_goal(self):
        return self.domain_goals

    def get_reward(self):
        return self._reward()

    def _reward(self):
        """
        Calculate reward based on task completion
        Returns:
            reward (float): Reward given by user.
        """
        if self.goal.task_complete():
            reward = 2.0 * self.max_turn
        elif self.agenda.is_empty():
            reward = -1.0 * self.max_turn
        else:
            reward = -1.0
        return reward

    @classmethod
    def _transform_usract_out(cls, action):
        # print('before transform', action)
        new_action = {}
        for act in action.keys():
            if '-' in act:
                if 'general' not in act:
                    (dom, intent) = act.split('-')
                    new_act = dom.capitalize() + '-' + intent.capitalize()
                    new_action[new_act] = []
                    for pairs in action[act]:
                        slot = REF_USR_DA_M[dom.capitalize()].get(pairs[0], None)
                        if pairs[0] == 'none' and pairs[1] == 'none':
                            new_action[new_act].append(['none', 'none'])
                        elif pairs[0] == 'choice' and pairs[1] == 'any':
                            new_action[new_act].append(['Choice', 'any'])
                        elif pairs[0] == 'NotBook' and pairs[1] == 'none':
                            new_action[new_act].append(['NotBook', 'none'])
                        elif slot is not None:
                            new_action[new_act].append([slot, pairs[1]])
                    if len(new_action[new_act]) == 0:
                        new_action.pop(new_act)
                    # new_action[new_act] = [[REF_USR_DA_M[dom.capitalize()].get(pairs[0], pairs[0]), pairs[1]] for pairs in action[act]]
                else:
                    new_action[act] = action[act]
            else:
                pass
        # print('after transform', new_action)
        return new_action

    @classmethod
    def _transform_sysact_in(cls, action):
        # print("sys in", action)
        new_action = {}
        if not isinstance(action, dict):
            logging.warning('illegal da: {}'.format(action))
            return new_action

        for act in action.keys():
            if not isinstance(act, str) or '-' not in act:
                logging.warning('illegal act: {}'.format(act))
                continue

            if 'general' not in act:
                (dom, intent) = act.lower().split('-')
                if dom in REF_SYS_DA_M.keys():
                    new_list = []
                    for pairs in action[act]:
                        if (not isinstance(pairs, list) and not isinstance(pairs, tuple)) or \
                                (len(pairs) < 2) or \
                                (not isinstance(pairs[0], str) or (not isinstance(pairs[1], str) and not isinstance(pairs[1], int))):
                            logging.warning('illegal pairs: {}'.format(pairs))
                            continue

                        if REF_SYS_DA_M[dom].get(pairs[0].lower(), None) is not None:
                            new_list.append([REF_SYS_DA_M[dom][pairs[0].lower()],
                                             cls._normalize_value(dom, intent, REF_SYS_DA_M[dom][pairs[0].lower()], pairs[1])])

                    if len(new_list) > 0:
                        new_action[act.lower()] = new_list
            else:
                new_action[act.lower()] = action[act]
        # print("sys in transformed", new_action)
        return new_action

    @classmethod
    def _normalize_value(cls, domain, intent, slot, value):
        if intent == 'request':
            return DEF_VAL_UNK

        if domain not in cls.stand_value_dict.keys():
            return value

        if slot not in cls.stand_value_dict[domain]:
            return value

        if slot in ['parking', 'internet'] and value == 'none':
            return 'yes'

        value_list = cls.stand_value_dict[domain][slot]
        low_value_list = [item.lower() for item in value_list]
        value_list = sorted(list(set(value_list)|set(low_value_list)))
        if value not in value_list:
            normalized_v = simple_fuzzy_match(value_list, value)
            if normalized_v is not None:
                return normalized_v
            # try some transformations
            cand_values = transform_value(value)
            for cv in cand_values:
                _nv = simple_fuzzy_match(value_list, cv)
                if _nv is not None:
                    return _nv
            if check_if_time(value):
                return value

            logging.debug('Value not found in standard value set: [%s] (slot: %s domain: %s)' % (value, slot, domain))
        return value


def transform_value(value):
    cand_list = []
    # a 's -> a's
    if " 's" in value:
        cand_list.append(value.replace(" 's", "'s"))
    # a - b -> a-b
    if " - " in value:
        cand_list.append(value.replace(" - ", "-"))
    return cand_list


def simple_fuzzy_match(value_list, value):
    # check contain relation
    v0 = ' '.join(value.split())
    v0N = ''.join(value.split())
    for val in value_list:
        v1 = ' '.join(val.split())
        if v0 in v1 or v1 in v0 or v0N in v1 or v1 in v0N:
            return v1
    value = value.lower()
    v0 = ' '.join(value.split())
    v0N = ''.join(value.split())
    for val in value_list:
        v1 = ' '.join(val.split())
        if v0 in v1 or v1 in v0 or v0N in v1 or v1 in v0N:
            return v1
    return None


def check_if_time(value):
    value = value.strip()
    match = re.search(r"(\d{1,2}:\d{1,2})", value)
    if match is None:
        return False
    groups = match.groups()
    if len(groups) <= 0:
        return False
    return True


def check_constraint(slot, val_usr, val_sys):
    try:
        if slot == 'arriveBy':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 < val2:
                return True
        elif slot == 'leaveAt':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 > val2:
                return True
        else:
            if val_usr != val_sys:
                return True
        return False
    except:
        return False


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal_generator: GoalGenerator):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Generator.
        """
        self.domain_goals = goal_generator.get_user_goal()

        self.domains = list(self.domain_goals['domain_ordering'])
        del self.domain_goals['domain_ordering']

        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}

            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK

    def set_user_goal(self, user_goal):
        """
        set new Goal given user goal generated by goal_generator.get_user_goal()
        Args:
            user_goal : user goal generated by GoalGenerator.
        """
        self.domain_goals = user_goal

        self.domains = list(self.domain_goals['domain_ordering'])
        del self.domain_goals['domain_ordering']

        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}

            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK


    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain]:
                reqt_vals = self.domain_goals[domain]['reqt'].values()
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        return False

            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return False
        return True

    def next_domain_incomplete(self):
        # request
        for domain in self.domains:
            # reqt
            if 'reqt' in self.domain_goals[domain]:
                requests = self.domain_goals[domain]['reqt']
                unknow_reqts = [key for (key, val) in requests.items() if val in NOT_SURE_VALS]
                if len(unknow_reqts) > 0:
                    return domain, 'reqt', ['name'] if 'name' in unknow_reqts else unknow_reqts

            # book
            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return domain, 'book', \
                           self.domain_goals[domain]['fail_book'] if 'fail_book' in self.domain_goals[domain].keys() else \
                               self.domain_goals[domain]['book']

        return None, None, None

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'


class Agenda(object):
    def __init__(self, goal: Goal):
        """
        Build a new agenda from goal
        Args:
            goal (Goal): User goal.
        """

        def random_sample(data, minimum=0, maximum=1000):
            return random.sample(data, random.randint(min(len(data), minimum), min(len(data), maximum)))

        self.CLOSE_ACT = 'general-bye'
        self.HELLO_ACT = 'general-greet'
        self.__cur_push_num = 0

        self.__stack = []

        # there is a 'bye' action at the bottom of the stack
        self.__push(self.CLOSE_ACT)

        for idx in range(len(goal.domains) - 1, -1, -1):
            domain = goal.domains[idx]

            # inform
            # first ask fail_info which return no result then ask info
            if 'fail_info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['fail_info'].keys(),
                                          len(goal.domain_goals[domain]['fail_info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['fail_info'][slot])
            elif 'info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['info'].keys(),
                                          len(goal.domain_goals[domain]['info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['info'][slot])

            self.__push(domain + '-inform', "none", "none")

        self.cur_domain = None

    def update(self, sys_action, goal: Goal):
        """
        update Goal by current agent action and current goal. { A' + G" + sys_action -> A" }
        Args:
            sys_action (dict): Preorder system action.s
            goal (Goal): User Goal
        """
        self.__cur_push_num = 0
        self._update_current_domain(sys_action, goal)

        for diaact in sys_action.keys():
            slot_vals = sys_action[diaact]
            if 'nooffer' in diaact:
                if self.update_domain(diaact, slot_vals, goal):
                    return
            elif 'nobook' in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if 'nooffer' in diaact or 'nobook' in diaact:
                continue

            slot_vals = sys_action[diaact]
            if 'booking' in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return
            elif 'general' in diaact:
                if self.update_general(diaact, slot_vals, goal):
                    return
            else:
                if self.update_domain(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if 'inform' in diaact or 'recommend' in diaact:
                for slot, val in sys_action[diaact]:
                    if slot == 'name':
                        self._remove_item(diaact.split('-')[0]+'-inform', 'choice')
            if 'booking' in diaact and self.cur_domain:
                g_book = self._get_goal_infos(self.cur_domain, goal)[-2]
                if len(g_book) == 0:
                    self._push_item(self.cur_domain + '-inform', "NotBook", "none")
            if 'OfferBook' in diaact:
                domain = diaact.split('-')[0]
                g_book = self._get_goal_infos(domain, goal)[-2]
                if len(g_book) == 0:
                    self._push_item(domain + '-inform', "NotBook", "none")

        self.post_process(goal)

    def post_process(self, goal: Goal):
        unk_dom, unk_type, data = goal.next_domain_incomplete()
        if unk_dom is not None:
            if unk_type == 'reqt' and not self._check_reqt_info(unk_dom) and not self._check_reqt(unk_dom):
                for slot in data:
                    self._push_item(unk_dom + '-request', slot, DEF_VAL_UNK)
            elif unk_type == 'book' and not self._check_reqt_info(unk_dom) and not self._check_book_info(unk_dom):
                for (slot, val) in data.items():
                    self._push_item(unk_dom + '-inform', slot, val)

    def update_booking(self, diaact, slot_vals, goal: Goal):
        """
        Handel Book-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        _, intent = diaact.split('-')
        domain = self.cur_domain

        isover = False
        if domain not in goal.domains:
            isover = False

        elif intent in ['book', 'inform']:
            isover = self._handle_inform(domain, intent, slot_vals, goal)

        elif intent in ['nobook']:
            isover = self._handle_nobook(domain, intent, slot_vals, goal)

        elif intent in ['request']:
            isover = self._handle_request(domain, intent, slot_vals, goal)

        return isover

    def update_domain(self, diaact, slot_vals, goal: Goal):
        """
        Handel Domain-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        domain, intent = diaact.split('-')

        isover = False
        if domain not in goal.domains:
            isover = False

        elif intent in ['inform', 'recommend', 'offerbook', 'offerbooked']:
            isover = self._handle_inform(domain, intent, slot_vals, goal)

        elif intent in ['request']:
            isover = self._handle_request(domain, intent, slot_vals, goal)

        elif intent in ['nooffer']:
            isover = self._handle_nooffer(domain, intent, slot_vals, goal)

        elif intent in ['select']:
            isover = self._handle_select(domain, intent, slot_vals, goal)

        return isover

    def update_general(self, diaact, slot_vals, goal: Goal):
        domain, intent = diaact.split('-')

        if intent == 'bye':
            # self.close_session()
            # return True
            pass
        elif intent == 'greet':
            pass
        elif intent == 'reqmore':
            pass
        elif intent == 'welcome':
            pass

        return False

    def close_session(self):
        """ Clear up all actions """
        self.__stack = []
        self.__cur_push_num = 0
        self.__push(self.CLOSE_ACT)

    def get_action(self, initiative=1):
        """
        get multiple acts based on initiative
        Args:
            initiative (int): number of slots , just for 'inform'
        Returns:
            action (dict): user diaact
        """
        # print(self)
        diaacts, slots, values = self.__pop(initiative)
        action = {}
        for (diaact, slot, value) in zip(diaacts, slots, values):
            if diaact not in action.keys():
                action[diaact] = []
            action[diaact].append([slot, value])

        self._setdefault_current_domain_by_usraction(action)

        return action

    def is_empty(self):
        """
        Is the agenda already empty
        Returns:
            (boolean): True for empty, False for not.
        """
        return len(self.__stack) <= 0

    @staticmethod
    def _my_value(value):
        new_value = value
        if value in NOT_SURE_VALS:
            new_value = '\"' + value + '\"'
        return new_value

    def _get_goal_infos(self, domain, goal: Goal):
        g_reqt = goal.domain_goals[domain].get('reqt', dict({}))
        g_info = goal.domain_goals[domain].get('info', dict({}))
        g_fail_info = goal.domain_goals[domain].get('fail_info', dict({}))
        g_book = goal.domain_goals[domain].get('book', dict({}))
        g_fail_book = goal.domain_goals[domain].get('fail_book', dict({}))
        return g_reqt, g_info, g_fail_info, g_book, g_fail_book

    def _handle_inform(self, domain, intent, slot_vals, goal: Goal):
        g_reqt, g_info, g_fail_info, g_book, g_fail_book = self._get_goal_infos(domain, goal)

        info_right = True
        for [slot, value] in slot_vals:
            if slot == 'time':
                if domain in ['train', 'restaurant']:
                    slot = 'duration' if domain == 'train' else 'time'
                else:
                    logging.warning('illegal booking slot: {}, domain: {}'.format(slot, domain))
                    continue

            # For multiple choices, add new intent to select one:
            if slot == 'choice' and value.strip().lower() not in ['0', 'zero', '1', 'one']:
                self._push_item(domain + '-inform', "choice", "any")

            if slot in g_reqt:
                if not self._check_reqt_info(domain):
                    self._remove_item(domain + '-request', slot)
                    g_reqt[slot] = self._my_value(value)

            elif slot in g_fail_info and value != g_fail_info[slot]:
                self._push_item(domain + '-inform', slot, g_fail_info[slot])
                info_right = False

            elif not g_fail_info and slot in g_info and check_constraint(slot, g_info[slot], value):
                self._push_item(domain + '-inform', slot, g_info[slot])
                info_right = False

            elif slot in g_fail_book and value != g_fail_book[slot]:
                self._push_item(domain + '-inform', slot, g_fail_book[slot])
                info_right = False

            elif not g_fail_book and slot in g_book and value != g_book[slot]:
                self._push_item(domain + '-inform', slot, g_book[slot])
                info_right = False

        if intent in ['book', 'offerbooked'] and info_right:
            # booked ok
            if 'booked' in goal.domain_goals[domain]:
                goal.domain_goals[domain]['booked'] = DEF_VAL_BOOKED
            # self._push_item('general-thank')

        return False

    def _handle_request(self, domain, intent, slot_vals, goal: Goal):
        g_reqt, g_info, g_fail_info, g_book, g_fail_book = self._get_goal_infos(domain, goal)
        for [slot, _] in slot_vals:
            if slot == 'time':
                if domain in ['train', 'restaurant']:
                    slot = 'duration' if domain == 'train' else 'time'
                else:
                    logging.warning('illegal booking slot: %s, slot: %s domain' % (slot, domain))
                    continue

            if slot in g_reqt:
                pass
            elif slot in g_fail_info:
                self._push_item(domain + '-inform', slot, g_fail_info[slot])
            elif not g_fail_info and slot in g_info:
                self._push_item(domain + '-inform', slot, g_info[slot])

            elif slot in g_fail_book:
                self._push_item(domain + '-inform', slot, g_fail_book[slot])
            elif not g_fail_book and slot in g_book:
                self._push_item(domain + '-inform', slot, g_book[slot])

            else:

                if domain == 'taxi' and (slot == 'destination' or slot == 'departure'):
                    places = [dom for dom in goal.domains[: goal.domains.index('taxi')] if
                              dom in ['attraction', 'hotel', 'restaurant', 'police', 'hospital']] # name will not appear in reqt
                    if len(places) >= 1 and slot == 'destination':
                        place_idx = -1
                    elif len(places) >= 2 and slot == 'departure':
                        place_idx = -2
                    else:
                        place_idx = None
                    if place_idx:
                        if goal.domain_goals[places[place_idx]]['info'].get('name', DEF_VAL_NUL) not in NOT_SURE_VALS:
                            place = goal.domain_goals[places[place_idx]]['info']['name']
                        # elif goal.domain_goals[places[place_idx]]['reqt'].get('address', DEF_VAL_NUL) not in NOT_SURE_VALS:
                        #     place = goal.domain_goals[places[place_idx]]['reqt']['address']
                        else:
                            place = "the " + places[place_idx]
                        self._push_item(domain + '-inform', slot, place)
                    else:
                        self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

                else:
                    # for those sys requests that are not in user goal
                    self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

        return False

    def _handle_nooffer(self, domain, intent, slot_vals, goal: Goal):
        g_reqt, g_info, g_fail_info, g_book, g_fail_book = self._get_goal_infos(domain, goal)
        if g_fail_info:
            # update info data to the stack
            for slot in g_info.keys():
                if (slot not in g_fail_info) or (slot in g_fail_info and g_fail_info[slot] != g_info[slot]):
                    self._push_item(domain + '-inform', slot, g_info[slot])

            # change fail_info name
            goal.domain_goals[domain]['fail_info_fail'] = goal.domain_goals[domain].pop('fail_info')
        elif g_reqt:
            self.close_session()
            return True
        return False

    def _handle_nobook(self, domain, intent, slot_vals, goal: Goal):
        g_reqt, g_info, g_fail_info, g_book, g_fail_book = self._get_goal_infos(domain, goal)
        if g_fail_book:
            # Discard fail_book data and update the book data to the stack
            for slot in g_book.keys():
                if (slot not in g_fail_book) or (slot in g_fail_book and g_fail_book[slot] != g_book[slot]):
                    self._push_item(domain + '-inform', slot, g_book[slot])

            # change fail_info name
            goal.domain_goals[domain]['fail_book_fail'] = goal.domain_goals[domain].pop('fail_book')
        elif 'booked' in goal.domain_goals[domain].keys():
            self.close_session()
            return True
        return False

    def _handle_select(self, domain, intent, slot_vals, goal: Goal):
        g_reqt, g_info, g_fail_info, g_book, g_fail_book = self._get_goal_infos(domain, goal)
        # delete Choice
        for slot, val in slot_vals:
            if slot == 'choice' and val.strip().lower() not in ['0', 'zero', '1', 'one']:
                self._push_item(domain + '-inform', "choice", "any")
        slot_vals = [[slot, val] for [slot, val] in slot_vals if slot != 'choice']

        if slot_vals:
            slot = slot_vals[0][0]

            if slot in g_fail_info:
                self._push_item(domain + '-inform', slot, g_fail_info[slot])
            elif not g_fail_info and slot in g_info:
                self._push_item(domain + '-inform', slot, g_info[slot])

            elif slot in g_fail_book:
                self._push_item(domain + '-inform', slot, g_fail_book[slot])
            elif not g_fail_book and slot in g_book:
                self._push_item(domain + '-inform', slot, g_book[slot])

            else:
                if not self._check_reqt_info(domain):
                    [slot, value] = random.choice(slot_vals)
                    self._push_item(domain + '-inform', slot, value)

                    if slot in g_reqt:
                        self._remove_item(domain + '-request', slot)
                        g_reqt[slot] = value
        return False

    def _update_current_domain(self, sys_action, goal: Goal):
        for diaact in sys_action.keys():
            domain, _ = diaact.split('-')
            if domain in goal.domains:
                self.cur_domain = domain

    def _setdefault_current_domain_by_usraction(self, usr_action):
        if self.cur_domain is None:
            for diaact in usr_action.keys():
                domain, _ = diaact.split('-')
                if domain in ['attraction', 'hotel', 'restaurant', 'taxi', 'train']:
                    self.cur_domain = domain

    def _remove_item(self, diaact, slot=DEF_VAL_UNK):
        for idx in range(len(self.__stack)):
            if 'general' in diaact:
                if self.__stack[idx]['diaact'] == diaact:
                    self.__stack.remove(self.__stack[idx])
                    break
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    self.__stack.remove(self.__stack[idx])
                    break

    def _push_item(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self._remove_item(diaact, slot)
        self.__push(diaact, slot, value)
        self.__cur_push_num += 1

    def _check_item(self, diaact, slot=None):
        for idx in range(len(self.__stack)):
            if slot is None:
                if self.__stack[idx]['diaact'] == diaact:
                    return True
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    return True
        return False

    def _check_reqt(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-request':
                return True
        return False

    def _check_reqt_info(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-inform' and self.__stack[idx]['slot'] not in BOOK_SLOT:
                return True
        return False

    def _check_book_info(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-inform' and self.__stack[idx]['slot'] in BOOK_SLOT:
                return True
        return False

    def __check_next_diaact_slot(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact'], self.__stack[-1]['slot']
        return None, None

    def __check_next_diaact(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact']
        return None

    def __push(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        if slot in ['people', 'day', 'area', 'pricerange']:
            for item in self.__stack:
                if item['slot'] == slot and item['value'] == value and random.random() < 0.3:
                    if slot == 'people':
                        item['value'] = 'the same'
                    elif slot == 'day':
                        item['value'] = 'the same day'
                    elif slot == 'pricerange':
                        item['value'] = 'in the same price range as the {}'.format(diaact.split('-')[0])
                    elif slot == 'area':
                        item['value'] = 'same area as the {}'.format(diaact.split('-')[0])
        self.__stack.append({'diaact': diaact, 'slot': slot, 'value': value})

    def __pop(self, initiative=1):
        diaacts = []
        slots = []
        values = []
        p_diaact, p_slot = self.__check_next_diaact_slot()
        if p_diaact.split('-')[1] == 'inform' and p_slot in BOOK_SLOT:
            for _ in range(10 if self.__cur_push_num == 0 else self.__cur_push_num):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact, next_slot = self.__check_next_diaact_slot()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            next_diaact.split('-')[1] != 'inform' or next_slot not in BOOK_SLOT:
                        break
                except Exception as e:
                    break
        else:
            if self.__cur_push_num == 0 or (all([self.__stack[-i-1]['value'] == DEF_VAL_DNC for i in
                                                 range(0, min(len(self.__stack), self.__cur_push_num))])):
                # pop more when only dontcare
                num2pop = initiative
            else:
                num2pop = self.__cur_push_num
            for _ in range(num2pop):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact = self.__check_next_diaact()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            (cur_diaact.split('-')[1] == 'request' and item['slot'] == 'name'):
                        break
                except Exception as e:
                    break

        return diaacts, slots, values

    def __str__(self):
        text = '\n-----agenda-----\n'
        text += '<stack top>\n'
        for item in reversed(self.__stack):
            text += str(item) + '\n'
        text += '<stack btm>\n'
        text += '-----agenda-----\n'
        return text


if __name__ == '__main__':
    import numpy as np
    import torch
    from pprint import pprint
    from convlab2.dialog_agent import PipelineAgent, BiSession
    from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
    from convlab2.policy.rule.multiwoz import RulePolicy
    from convlab2.nlg.template.multiwoz.nlg import TemplateNLG
    from convlab2.dst.rule.multiwoz.dst import RuleDST
    from convlab2.nlu.jointBERT.multiwoz.nlu import BERTNLU

    seed = 41
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #
    # sys_nlu = BERTNLU()
    # sys_dst = RuleDST()
    # sys_policy = RulePolicy()
    # sys_nlg = TemplateNLG(is_user=False)
    # sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
    #                    model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    # user_dst = None
    # user_policy = RulePolicy(character='usr')
    # user_nlg = TemplateNLG(is_user=True)
    # user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    # evaluator = MultiWozEvaluator()
    # sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)




    user_policy = UserPolicyAgendaMultiWoz()
    #
    sys_policy = RulePolicy(character='sys')
    #
    user_nlg = TemplateNLG(is_user=True, mode='manual')
    sys_nlg = TemplateNLG(is_user=False, mode='manual')
    #
    dst = RuleDST()
    #
    # user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
    #                    model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    #
    goal_generator = GoalGenerator()
    # while True:
    #     goal = goal_generator.get_user_goal()
    #     if 'restaurant' in goal['domain_ordering'] and 'hotel' in goal['domain_ordering']:
    #         break
    # # pprint(goal)
    user_goal = {'domain_ordering': ('hotel', 'attraction'),
                 'train': {
                     'info': {'arriveBy': '16:00',
                              'day': 'monday',
                              'departure': 'cambridge',
                              'destination': 'stansted airport'},
                     'book': {'people': 2}, 'booked': '?'
                 },
                 'attraction': {
                     'info': {'type': 'museum'},
                     'reqt': ['phone']
                 },
                 'hotel': {
                           'info': {'internet': 'yes',
                                    'parking': 'yes',
                                    'stars': '4',
                                    'type': 'hotel'},
                           'reqt': ['postcode']},
                 'restaurant': {'info': {'area': 'centre',
                                         'food': 'portuguese',
                                         'pricerange': 'cheap'},
                                'fail_info': {'area': 'centre',
                                         'food': 'portuguese',
                                         'pricerange': 'expensive'},
                                'reqt': ['postcode']},
                 'taxi': {'info': {'arriveBy': '13:00'}, 'reqt': ['car type', 'phone']}}
    # # user_goal = goal
    goal = Goal(goal_generator)
    goal.set_user_goal(user_goal)
    #
    # user_policy.init_session(ini_goal=goal)
    # sys_policy.init_session()
    #
    # goal = user_policy.get_goal()
    #
    # pprint(goal)

    sys_response = ''
    # sess.init_session(ini_goal=goal)
    user_policy.init_session(ini_goal=goal)
    print('init goal:')
    # pprint(user_policy.get_goal())
    # pprint(user_agent.policy.get_goal())
    # pprint(sess.evaluator.goal)
    # print('-' * 50)
    # for i in range(20):
    #     sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    #     print('user:', user_response)
    #     print('sys:', sys_response)
    #     print()
    #     if session_over is True:
    #         break
    # print('task success:', sess.evaluator.task_success())
    # print('book rate:', sess.evaluator.book_rate())
    # print('inform precision/recall/f1:', sess.evaluator.inform_F1())
    # print('-' * 50)
    # print('final goal:')
    # pprint(sess.evaluator.goal)
    # print('=' * 100)

    history = []
    # user_utt = user_agent.response('')
    # print(user_utt)
    # user_utt = 'I need a restaurant . It just needs to be expensive . I am also in the market for a new restaurant . Is there something in the centre of town ? Do you have portuguese food ?'
    # # history.append(['user', user_utt])
    # sys_agent.dst.state['belief_state']['restaurant']['semi']['food'] = 'portuguese'
    # sys_utt = sys_agent.response(user_utt)
    # pprint(sys_agent.dst.state)
    # print(sys_utt)
    # sys_utt = "I have n't found any in the centre. I am unable to find any portuguese restaurants in town ."
    # # history.append(['user', user_utt])
    #
    # user_utt = user_agent.response(sys_utt)
    # print(user_utt)
    # user_utt = "It just needs to be cheap ."
    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)
    # sys_utt = "It is in the centre area . They serve portuguese . Would you like to try nandos city centre ? They are in the cheap price range . I will book it for you and get a reference number ?"
    #
    # user_utt = user_agent.response(sys_utt)
    # print(user_utt)
    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)
    #
    # user_utt = user_agent.response(sys_utt)
    # print(user_utt)
    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)
    #
    # user_utt = user_agent.response(sys_utt)
    # print(user_utt)
    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)

    #
    print(user_policy.agenda)
    user_act = user_policy.predict([])
    print(user_act)
    user_utt = user_nlg.generate(user_act)
    print(user_utt)
    history.append(['user', user_utt])
    state = dst.state
    state['user_action'] = user_act
    dst.update(user_act)
    # pprint(state)
    sys_act = sys_policy.predict(state)
    sys_utt = sys_nlg.generate(sys_act)
    # sys_act.append(["Request", "Restaurant", "Price", "?"])
    # sys_act = [['Request', 'Hotel', 'Area', '?'], ['Request', 'Hotel', 'Stars', '?']]
    sys_act = [['Inform', 'Hotel', 'Post', 'pe296fl']]
    print(sys_act)
    history.append(['sys', user_utt])

    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)
    #
    user_act = user_policy.predict(sys_act)
    print(user_act)
    user_utt = user_nlg.generate(user_act)
    print(user_utt)
    history.append(['user', user_utt])
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [['Inform', 'Hotel', 'Choice', '3']]
    # print(sys_act)
    sys_act = [
        ['Inform', 'Hotel', 'Post', 'pe296fl']
    ]
    print(sys_act)
    # sys_utt = sys_agent.response(user_utt)
    # print(sys_utt)
    # sys_utt = 'The arrive time is 15:08 . The train will be departing from cambridge . The booking is for arriving in stansted airport . TR6936 will be your perfect fit . How about 14:40 will that work for you ?'
    # history.append(['sys', user_utt])
    #
    #
    # sys_act = user_nlu.predict(sys_utt, history)
    # print(sys_act)
    user_act = user_policy.predict(sys_act)
    print(user_act)
    user_utt = user_nlg.generate(user_act)
    print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    sys_act = [['Request', 'Hotel', 'Price', '?'], ['Request', 'Attraction', 'Price', '?']]
    print(sys_act)
    # #
    user_act = user_policy.predict(sys_act)
    print(user_act)
    user_utt = user_nlg.generate(user_act)
    print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [["Reqmore", "General", "none", "none"]]
    # print(sys_act)
    # #
    # user_act = user_policy.predict(sys_act)
    # print(user_act)
    # user_utt = user_nlg.generate(user_act)
    # print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [["Inform", "Hotel", "Parking", "none"]]
    # print(sys_act)
    #
    # user_act = user_policy.predict(sys_act)
    # print(user_act)
    # user_utt = user_nlg.generate(user_act)
    # print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [["Request", "Booking", "people", "?"]]
    # print(sys_act)
    #
    # user_act = user_policy.predict(sys_act)
    # print(user_act)
    # user_utt = user_nlg.generate(user_act)
    # print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [["Inform", "Hotel", "Post", "233"], ["Book", "Booking", "none", "none"]]
    # print(sys_act)
    #
    # user_act = user_policy.predict(sys_act)
    # print(user_act)
    # user_utt = user_nlg.generate(user_act)
    # print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # sys_act = [["Request", "Taxi", "Dest", "?"], ["Request", "Taxi", "Depart", "?"]]
    # print(sys_act)
    #
    # user_act = user_policy.predict(sys_act)
    # print(user_act)
    # user_utt = user_nlg.generate(user_act)
    # print(user_utt)
    # state = dst.state
    # state['user_action'] = user_act
    # dst.update(user_act)
    # # pprint(state)
    # sys_act = sys_policy.predict(state)
    # # sys_act = [["Request", "Taxi", "Destination", "?"], ["Request", "Taxi", "Departure", "?"]]
    # print(sys_act)
