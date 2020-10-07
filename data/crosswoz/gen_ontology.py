import json
from zipfile import ZipFile

import re

ontology = {
    "景点": {
        "名称": set(),
        "门票": set(),
        "游玩时间": set(),
        "评分": set(),
        "周边景点": set(),
        "周边餐馆": set(),
        "周边酒店": set(),
    },
    "餐馆": {
        "名称": set(),
        "推荐菜": set(),
        "人均消费": set(),
        "评分": set(),
        "周边景点": set(),
        "周边餐馆": set(),
        "周边酒店": set(),
    },
    "酒店": {
        "名称": set(),
        "酒店类型": set(),
        "酒店设施": set(),
        "价格": set(),
        "评分": set(),
        "周边景点": set(),
        "周边餐馆": set(),
        "周边酒店": set(),
    },
    "地铁": {
        "出发地": set(),
        "目的地": set(),
    },
    "出租": {
        "出发地": set(),
        "目的地": set(),
    }
}

if __name__ == '__main__':
    pattern = re.compile('. .+')
    for split in ['train', 'val', 'test', 'dstc9_data']:
        print(split)
        with ZipFile(f'{split}.json.zip', 'r') as zipfile:
            with zipfile.open(f'{split}.json', 'r') as f:
                data = json.load(f)

        for dialog in data.values():
            for turn in dialog['messages']:
                if turn['role'] == 'sys':
                    state = turn['sys_state_init']
                    for domain_name, domain in state.items():
                        for slot_name, value in domain.items():

                            if slot_name == 'selectedResults':
                                continue
                            else:
                                value = value.replace('\t', ' ').strip()
                                if not value:
                                    continue
                                values = ontology[domain_name][slot_name]
                                if slot_name in ['酒店设施', '推荐菜']:
                                    # deal with values contain bothering space like "早 餐 服 务   无 烟 房"
                                    if pattern.match(value):
                                        print(value)
                                        value = value.replace('   ', ';').replace(' ', '').replace(';', ' ')
                                        print(value)
                                    for v in value.split(' '):
                                        if v:
                                            values.add(v)
                                elif value and value not in values:
                                    # if ',' in value or '，' in value or ' ' in value:
                                        # print(value, slot_name)
                                    values.add(value)

    for domain in ontology.values():
        for slot_name, values in domain.items():
            domain[slot_name] = list(values)

    with open('ontology.json', 'w') as f:
        json.dump(ontology, f, indent=4, ensure_ascii=False)
