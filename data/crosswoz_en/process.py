import json
import zipfile
from copy import deepcopy


def pop_sr(state):
    state = deepcopy(state)
    state.pop('selectedResults')
    return state

def process_data():
    for split in ['train', 'val', 'test', 'human_val']:
        print(split)
        data = json.load(zipfile.ZipFile(f'{split}.json.zip').open(f'{split}.json'))
        for dialog_id, dialog in data.items():
            dialog_data = []
            turns = dialog['messages']
            selected_results = {domain_name: [] for domain_name in turns[1]['sys_state_init']}
            for i in range(0, len(turns), 2):
                dialog_state = turns[i + 1]['sys_state_init']
                for domain_name, state in dialog_state.items():
                    sys_selected_results = turns[i + 1]['sys_state'][domain_name]['selectedResults']
                    # if state has changed compared to previous sys state
                    state_change = i == 0 or pop_sr(state) != pop_sr(turns[i - 1]['sys_state'][domain_name])
                    # clear the outdated previous selected results if state has been updated
                    if state_change:
                        selected_results[domain_name].clear()
                    if not state.get('name', 'something nonempty') and len(selected_results[domain_name]) == 1:
                        state['name'] = selected_results[domain_name][0]
                    if state_change and sys_selected_results:
                        selected_results[domain_name] = sys_selected_results
        zipfile.ZipFile(f'{split}.json.zip', 'w').open(f'{split}.json', 'w').write(json.dumps(data, indent=4, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    ontology = json.load(open('ontology-data.json'))
    new_ontology = {}
    for domain_name, domain in ontology.items():
        for slot_name, values in domain.items():
            new_ontology[f'{domain_name}-{slot_name}'] = values
    json.dump(new_ontology, open('ontology.json', 'w'), indent=4, ensure_ascii=False)
