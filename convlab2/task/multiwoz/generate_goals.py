"""
generate user goal for collecting new multiwoz data
"""

from convlab2.task.multiwoz.goal_generator import GoalGenerator
from convlab2.util.file_util import read_zipped_json
import random
import numpy as np
import json
import datetime
from pprint import pprint


def extract_slot_combination_from_goal(goal):
    domains = ['attraction', 'hotel', 'restaurant', 'police', 'hospital', 'taxi', 'train']
    serialized_goal = []
    for domain in goal:
        if domain in domains:
            for scope, content in goal[domain].items():
                if content:
                    # if isinstance(content, dict):
                    #     for slot, value in content.items():
                    #         serialized_goal.append("{}-{}-{}-{}".format(domain, scope, slot, value))
                    # else:
                    #     for slot in content:
                    #         serialized_goal.append("{}-{}-{}".format(domain, scope, slot))
                    for slot in content:
                        serialized_goal.append("{}-{}-{}".format(domain, scope, slot))
    return sorted(serialized_goal)


def test_generate_overlap(total_num=1000, seed=42, output_file='goal.json'):
    train_data = read_zipped_json('../../../data/multiwoz/train.json.zip', 'train.json')
    train_serialized_goals = []
    for d in train_data:
        train_serialized_goals.append(extract_slot_combination_from_goal(train_data[d]['goal']))

    test_data = read_zipped_json('../../../data/multiwoz/test.json.zip', 'test.json')
    test_serialized_goals = []
    for d in test_data:
        test_serialized_goals.append(extract_slot_combination_from_goal(test_data[d]['goal']))

    overlap = 0
    for serialized_goal in test_serialized_goals:
        if serialized_goal in train_serialized_goals:
            overlap += 1
    print(len(train_serialized_goals), len(test_serialized_goals), overlap) # 8434 1000 430

    random.seed(seed)
    np.random.seed(seed)
    goal_generator = GoalGenerator()
    goals = []
    avg_domains = []
    serialized_goals = []
    while len(goals) < total_num:
        goal = goal_generator.get_user_goal()
        # pprint(goal)
        if 'police' in goal['domain_ordering']:
            no_police = list(goal['domain_ordering'])
            no_police.remove('police')
            goal['domain_ordering'] = tuple(no_police)
            del goal['police']
        try:
            message = goal_generator.build_message(goal)[1]
        except:
            continue
        # print(message)
        avg_domains.append(len(goal['domain_ordering']))
        goals.append({
            "goals": [],
            "ori_goals": goal,
            "description": message,
            "timestamp": str(datetime.datetime.now()),
            "ID": len(goals)
        })
        serialized_goals.append(extract_slot_combination_from_goal(goal))
        if len(serialized_goals) == 1:
            print(serialized_goals)
    overlap = 0
    for serialized_goal in serialized_goals:
        if serialized_goal in train_serialized_goals:
            overlap += 1
    print(len(train_serialized_goals), len(serialized_goals), overlap) # 8434 1000 199


def generate(total_num=1000, seed=42, output_file='goal.json'):
    random.seed(seed)
    np.random.seed(seed)
    goal_generator = GoalGenerator()
    goals = []
    avg_domains = []
    while len(goals) < total_num:
        goal = goal_generator.get_user_goal()
        # pprint(goal)
        if 'police' in goal['domain_ordering']:
            no_police = list(goal['domain_ordering'])
            no_police.remove('police')
            goal['domain_ordering'] = tuple(no_police)
            del goal['police']
        try:
            message = goal_generator.build_message(goal)[1]
        except:
            continue
        # print(message)
        avg_domains.append(len(goal['domain_ordering']))
        goals.append({
            "goals": [],
            "ori_goals": goal,
            "description": message,
            "timestamp": str(datetime.datetime.now()),
            "ID": len(goals)
        })
    print('avg domains:', np.mean(avg_domains)) # avg domains: 1.846
    json.dump(goals, open(output_file, 'w'), indent=4)


if __name__ == '__main__':
    generate(output_file='goal20200629.json')
