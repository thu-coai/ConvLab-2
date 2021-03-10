import json
import os
from copy import deepcopy

special_values = ['dontcare', '']


def check_ontology(name):
    """
    ontology: {
        "domains": {
            domain name: {
                "description": domain description,
                "slots": {
                    slot name: {
                        "description": slot description
                        // possible_values is empty iff is_categorical is False
                        "is_categorical": is_categorical,
                        "possible_values": [possible_values...]
                    }
                }
            }
        },
        "intents": {
            intent name: {
                "description": intent description
            }
        },
        "binary_dialogue_act": {
            [
                {
                    "intent": intent name,
                    "domain": domain name
                    "slot": slot name,
                    "value": some value
                }
            ]
        }
        "state": {
            domain name: {
                slot name: ""
            }
        }
    }
    """
    global special_values

    ontology_file = os.path.join(f'{name}', 'ontology.json')
    assert os.path.exists(ontology_file), f'ontology file should named {ontology_file}'
    ontology = json.load(open(ontology_file))
    
    # record issues in ontology
    descriptions = {
        # if each domain has a description
        "domains": True,
        "slots": True,
        "intents": True,
    }
    for domain_name, domain in ontology['domains'].items():
        if not domain['description']:
            descriptions["domains"] = False
        # if not domain_name in ontology['state']:
        #     print(f"domain '{domain_name}' not found in state")
        for slot_name, slot in domain["slots"].items():
            if not slot["description"]:
                descriptions["slots"] = False
            if slot["is_categorical"]:
                assert slot["possible_values"]
                slot['possible_values'] = list(map(str.lower, slot['possible_values']))
                for value in special_values:
                    assert value not in slot['possible_values'], f'ONTOLOGY\tspecial value `{value}` should not present in possible values'

    for intent_name, intent in ontology["intents"].items():
        if not intent["description"]:
            descriptions["intents"] = False

    binary_dialogue_acts = set()
    for bda in ontology['binary_dialogue_act']:
        assert bda['intent'] is None or bda["intent"] in ontology['intents'], f'ONTOLOGY\tintent undefined intent in binary dialog act: {bda}'
        binary_dialogue_acts.add(tuple(bda.values()))
    ontology['bda_set'] = binary_dialogue_acts

    assert 'state' in ontology, 'ONTOLOGY\tno state'
    redundant_value = False
    for domain_name, domain in ontology['state'].items():
        assert domain_name in ontology['domains']
        for slot_name, value in domain.items():
            assert slot_name in ontology['domains'][domain_name]['slots']
            if value:
                redundant_value = True

    if redundant_value:
        print('ONTOLOGY: redundant value description in state')

    # print('description existence:', descriptions, '\n')
    for description, value in descriptions.items():
        if not value:
            print(f'description of {description} is incomplete')
    return ontology


def check_data(name, ontology):
    global special_values

    from zipfile import ZipFile
    data_file = os.path.join(f'{name}', 'data.zip')
    if not os.path.exists(data_file):
        print('cannot find data.zip')
        return

    print('loading data')
    with ZipFile(data_file) as zipfile:
        with zipfile.open('data.json', 'r') as f:
            data = json.load(f)

    all_id = set()
    splits = ['train', 'val', 'test']
    da_values = 0
    da_matches = 0
    state_values = 0
    state_matches = 0
    distances = []
    stat_keys = ['dialogues', 'utterances', 'tokens', 'domains']
    stat = {
        split: {
            key: 0 for key in stat_keys
        } for split in splits
    }

    # present for both non-categorical or categorical

    for dialogue in data:
        dialogue_id = dialogue['dialogue_id']
        assert isinstance(dialogue_id, str), '`dialogue_id` is expected to be str type'
        dialogue_id = str(dialogue_id)

        assert dialogue['dataset'] == name, f'{dialogue_id}\tinconsistent dataset name: {dialogue["dataset"]}'

        split = dialogue['data_split']
        assert split in splits, f'unknown split: `{split}`'
        cur_stat = stat[split]
        cur_stat['dialogues'] += 1
        try:
            prefix, num = dialogue_id.split('_')
            assert prefix == name
            int(num)    # try converting to int
        except:
            print(f'{dialogue_id}\twrong dialogue id format: {dialogue_id}')
            raise Exception
        assert dialogue_id not in all_id, f'multiple dialogue id: {dialogue_id}'
        all_id.add(dialogue_id)

        cur_domains = dialogue['domains']
        assert isinstance(cur_domains, list), f'{dialogue_id}\t`domains` is expected to be list type, '
        assert len(set(cur_domains)) == len(cur_domains), f'{dialogue_id}\trepeated domains'
        cur_stat['domains'] += len(cur_domains)
        cur_domains = set(cur_domains)
        for domain_name in cur_domains:
            assert domain_name in ontology['domains'], f'{dialogue_id}\tundefined current domain: {domain_name}'

        turns = dialogue['turns']
        cur_stat['utterances'] += len(turns)
        assert turns, f'{dialogue_id}\tempty turn'

        assert turns[0]['speaker'] == 'user', f'{dialogue_id}\tnot start with user role'
        if ontology['state']:
            # update cur_state with state_update every turn, and compare it with state annotation
            cur_state = {
                domain_name: deepcopy(ontology['state'][domain_name]) for domain_name in cur_domains
            }
        # check dialog act
        for turn_id, turn in enumerate(turns):
            assert turn['speaker'] in ['user', 'system'], f'{dialogue_id}:{turn_id}\tunknown speaker value: {turn["speaker"]}'
            assert turn_id == turn['utt_idx'], f'{dialogue_id}:{turn_id}\twrong utt_idx'
            if turn_id > 0:
                assert turns[turn_id - 1]['speaker'] != turn['speaker'], f'{dialogue_id}:{turn_id}\tuser and system should speak alternatively'

            utterance = turn['utterance']
            cur_stat['tokens'] += len(utterance.strip().split(' '))
            dialogue_acts = turn['dialogue_act']

            # check domain-slot-value
            # prefix: error prefix
            def check_dsv(domain_name, slot_name, value, categorical, prefix):
                assert domain_name in cur_domains or domain_name == 'booking', f'{prefix}\t{domain_name} not presented in current domains'
                domain = ontology['domains'][domain_name]
                assert slot_name in domain['slots'], f'{prefix}\t{slot_name} not presented in domain {domain_name}'
                slot = domain['slots'][slot_name]
                if categorical:
                    assert slot['is_categorical'], f'{prefix}\t{domain_name}-{slot_name} is not categorical'
                    value = value.lower()
                    assert value in special_values or value in slot['possible_values'], f'{prefix}\t`{value}` not presented in possible values of' \
                                                             f' {domain_name}-{slot_name}: {slot["possible_values"]}'
                else:
                    assert not slot['is_categorical'], f'{prefix}\t{domain_name}-{slot_name} is not non-categorical'

            def check_da(da, categorical):
                assert da['intent'] in ontology['intents'], f'{dialogue_id}:{turn_id}\tundefined intent {da["intent"]}'
                check_dsv(da['domain'], da['slot'], da['value'], categorical, f'{dialogue_id}:{turn_id}:da')

            for da in dialogue_acts['categorical']:
                check_da(da, True)
            for da in dialogue_acts['non-categorical']:
                check_da(da, False)
                # values only match after .strip() in some case, it's the issue of pre-processing
                if da['value'] not in special_values:
                    da_values += 1
                    assert 'start' in da and 'end' in da or 'start' not in da and 'end' not in da, \
                        f'{dialogue_id}:{turn_id}\tstart and end field in da should both present or neither not present'
                    if 'start' in da:
                        value = utterance[da['start']:da['end']]
                        if da['value'].lower() == value.lower():
                            da_matches += 1

            for da in dialogue_acts['binary']:
                assert tuple(da.values()) in ontology['bda_set'], f'{dialogue_id}:{turn_id}\tbinary dialog act {da} not present in ontology'
                # do not check domain-slot-value in binary dialogue acts

            if turn['speaker'] == 'user':
                assert 'state' in turn and 'state_update' in turn, f"{dialogue_id}:{turn_id}\tstate and state_update must present in user's role"
                state_update = turn['state_update']

                def apply_update(update, categorical):
                    domain_name = update['domain']
                    slot_name = update['slot']
                    value = update['value']
                    check_dsv(domain_name, slot_name, value, categorical, f'{dialogue_id}:{turn_id}:state_update')
                    cur_state[domain_name][slot_name] = value
                if ontology['state']:
                    for update in state_update['categorical']:
                        apply_update(update, True)
                    for update in state_update['non-categorical']:
                        apply_update(update, False)
                        value = update['value']
                        if value not in special_values:
                            state_values += 1
                            if 'utt_idx' in update:
                                if turns[update['utt_idx']]['utterance'][update['start']:update['end']].lower() == update['value']:
                                    state_matches += 1
                                else:
                                    print(turns[update['utt_idx']]['utterance'][update['start']:update['end']].strip())
                                    print(update['value'])
                                    pass

                    assert cur_state == turn['state'], f'{dialogue_id}:{turn_id}:state_update incorrect state or state update calculation'

            else:
                assert 'state' not in turn or 'state_update' in turn, f"{dialogue_id}:{turn_id}\tstate or state_update cannot present in system's role"

        assert turns[-1]['speaker'] == 'user', f'{dialogue_id} dialog must end with user role'

    if da_values:
        print('da values match rate:    {:.3f}'.format(da_matches * 100 / da_values))
    if state_values:
        print('state values match rate: {:.3f}'.format(state_matches * 100 / state_values))

    all_stat = {key: 0 for key in stat_keys}
    for key in stat_keys:
        all_stat[key] = sum(stat[split][key] for split in splits)
    stat['all'] = all_stat

    for split in splits + ['all']:
        cur_stat = stat[split]
        if cur_stat['dialogues']:
            cur_stat['avg_utt'] = round(cur_stat['utterances'] / cur_stat['dialogues'], 2)
            cur_stat['avg_tokens'] = round(cur_stat['tokens'] / cur_stat['utterances'], 2)
            cur_stat['avg_domains'] = round(cur_stat.pop('domains') / cur_stat['dialogues'], 2)
        else:
            del stat[split]
    print(f'domains: {len(ontology["domains"])}')
    print(json.dumps(stat, indent=4))
    if state_matches:
        for dis, cnt in enumerate(distances):
            print(cnt)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="evaluate pre-processed datasets")
    parser.add_argument('datasets', metavar='dataset_name', nargs='*', help='dataset names to be evaluated')
    parser.add_argument('--all', action='store_true', help='evaluate all datasets')
    parser.add_argument('--no-int', action='store_true', help='not interrupted by exception')
    parser.add_argument('--preprocess', '-p', action='store_true', help='run preprocess automatically')
    args = parser.parse_args()

    if args.all:
        datasets = list(filter(os.path.isdir, os.listdir()))
    else:
        datasets = args.datasets
    if not datasets:
        print('no dataset specified')
        parser.print_help()
        exit(1)

    print('datasets to be evaluated:', datasets)

    fail = []

    for name in datasets:
        try:
            print('')
            if not os.path.isdir(name):
                print(f'dataset {name} not found')
                continue

            print(f'checking {name}')
            preprocess_file = os.path.join(f'{name}', 'preprocess.py')
            if not os.path.exists(preprocess_file):
                print('no preprocess.py')
                if args.preprocess:
                    print(f'skip evaluation of {name}')
                    continue
            if args.preprocess:
                print('pre-processing')

                os.chdir(name)
                import importlib
                preprocess = importlib.import_module(f'{name}.preprocess')
                preprocess.preprocess()
                os.chdir('..')

            ontology = check_ontology(name)
            check_data(name, ontology)
        except Exception as e:
            if args.no_int:
                fail.append(name)
            else:
                raise e

    if not fail:
        print('all datasets passed test')
    else:
        print('failed dataset(s):', fail)
