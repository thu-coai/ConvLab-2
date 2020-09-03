"""
"""
import json
import os
import random
from copy import deepcopy


class Database(object):
    def __init__(self):
        super(Database, self).__init__()
        # loading databases
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
        self.dbs = {}
        for domain in domains:
            with open(os.path.join(os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    'data/multiwoz/db/{}_db.json'.format(domain))) as f:
                self.dbs[domain] = json.load(f)

    def query(self, domain, constraints, ignore_open=False):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state"""
        # query the db
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
            'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
            'taxi_phone': ''.join([str(random.randint(1, 9)) for _ in range(11)])}]
        if domain == 'police':
            return deepcopy(self.dbs['police'])
        if domain == 'hospital':
            department = None
            for key, val in constraints:
                if key == 'department':
                    department = val
            if not department:
                return deepcopy(self.dbs['hospital'])
            else:
                return [deepcopy(x) for x in self.dbs['hospital'] if x['department'].lower() == department.strip().lower()]
        for ele in constraints:
            if ele[0] == 'area' and ele[1] == 'center':
                ele[1] = 'centre'

        found = []
        for i, record in enumerate(self.dbs[domain]):
            for key, val in constraints:
                if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    try:
                        record_keys = [k.lower() for k in record]
                        if key.lower() not in record_keys:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        # elif ignore_open and key in ['destination', 'departure', 'name']:
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        else:
                            if val.strip().lower() != record[key].strip().lower():
                                break
                    except:
                        continue
            else:
                res = deepcopy(record)
                res['Ref'] = '{0:08d}'.format(i)
                found.append(res)

        return found


if __name__ == '__main__':
    db = Database()
    print(db.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arriveBy', '11:15']]))
