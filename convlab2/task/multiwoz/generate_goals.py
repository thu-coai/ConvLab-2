"""
generate user goal for collecting new multiwoz data
"""

from convlab2.task.multiwoz.goal_generator import GoalGenerator
import random
import numpy as np
import json
import datetime
from pprint import pprint


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
