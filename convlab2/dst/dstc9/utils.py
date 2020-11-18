import json
import os
import zipfile

from convlab2 import DATA_ROOT


def get_subdir(subtask):
    subdir = 'multiwoz_zh' if subtask == 'multiwoz' else 'crosswoz_en'
    return subdir


def prepare_data(subtask, split, data_root=DATA_ROOT, correct_name_label=False):
    data_dir = os.path.join(data_root, get_subdir(subtask))
    zip_filename = os.path.join(data_dir, f'{split}.json.zip')
    test_data = json.load(zipfile.ZipFile(zip_filename).open(f'{split}.json'))
    data = {}
    if subtask == 'multiwoz':
        for dialog_id, dialog in test_data.items():
            dialog_data = []
            turns = dialog['log']
            for i in range(0, len(turns), 2):
                sys_utt = turns[i - 1]['text'] if i else None
                user_utt = turns[i]['text']
                dialog_state = {}
                for domain_name, domain in turns[i + 1]['metadata'].items():
                    if domain_name in ['警察机关', '医院', '公共汽车']:
                        continue
                    state = {}
                    for slots in domain.values():
                        for slot_name, value in slots.items():
                            state[slot_name] = value
                    dialog_state[domain_name] = state
                dialog_data.append((sys_utt, user_utt, dialog_state))
            data[dialog_id] = dialog_data
    else:
        for dialog_id, dialog in test_data.items():
            dialog_data = []
            turns = dialog['messages']
            if correct_name_label:
                selected_results = {domain_name: [] for domain_name in turns[1]['sys_state_init']}
            for i in range(0, len(turns), 2):
                sys_utt = turns[i - 1]['content'] if i else None
                user_utt = turns[i]['content']
                dialog_state = {}
                for domain_name, state in turns[i + 1]['sys_state_init'].items():
                    if correct_name_label:
                        state.pop('selectedResults')
                        sys_selected_results = turns[i + 1]['sys_state'][domain_name].pop('selectedResults')
                        # if state has changed compared to previous sys state
                        state_change = i == 0 or state != turns[i - 1]['sys_state'][domain_name]
                        # clear the outdated previous selected results if state has been updated
                        if state_change:
                            selected_results[domain_name].clear()
                        if not state.get('name', 'something nonempty') and len(selected_results[domain_name]) == 1:
                            state['name'] = selected_results[domain_name][0]
                        dialog_state[domain_name] = state
                        if state_change and sys_selected_results:
                            selected_results[domain_name] = sys_selected_results
                    else:
                        selected_results = state.pop('selectedResults')
                        if selected_results and 'name' in state and not state['name']:
                            state['name'] = selected_results
                        dialog_state[domain_name] = state
                dialog_data.append((sys_utt, user_utt, dialog_state))
            data[dialog_id] = dialog_data

    return data


def extract_gt(test_data):
    gt = {
        dialog_id: [state for _, _, state in turns]
        for dialog_id, turns in test_data.items()
    }
    return gt


# for unifying values with the same meaning to the same expression
def unify_value(value, subtask):
    if isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = unify_value(v, subtask)
        return value

    value = value.lower()
    value = {
        'multiwoz': {
            '未提及': '',
            'none': '',
            '是的': '有',
            '不是': '没有',
        },
        'crosswoz': {
            'none': '',
            'free admission': 'free',
        }
    }[subtask].get(value, value)

    return ''.join(value.strip().split())


def eval_states(gt, pred, subtask):
    def exception(description, **kargs):
        ret = {
            'status': 'exception',
            'description': description,
        }
        for k, v in kargs.items():
            ret[k] = v
        return ret, None
    errors = [['dialog id', 'turn id', 'domain name', 'slot name', 'ground truth', 'predict']]

    joint_acc, joint_tot = 0, 0
    slot_acc, slot_tot = 0, 0
    tp, fp, fn = 0, 0, 0
    for dialog_id, gt_states in gt.items():
        if dialog_id not in pred:
            return exception('some dialog not found', dialog_id=dialog_id)

        pred_states = pred[dialog_id]
        if len(gt_states) != len(pred_states):
            return exception(f'turns number incorrect, {len(gt_states)} expected, {len(pred_states)} found', dialog_id=dialog_id)

        for turn_id, (gt_state, pred_state) in enumerate(zip(gt_states, pred_states)):
            joint_tot += 1
            turn_result = True
            for domain_name, gt_domain in gt_state.items():
                if domain_name not in pred_state:
                    return exception('domain missing', dialog_id=dialog_id, turn_id=turn_id, domain=domain_name)

                pred_domain = pred_state[domain_name]
                for slot_name, gt_value in gt_domain.items():
                    if slot_name not in pred_domain:
                        return exception('slot missing', dialog_id=dialog_id, turn_id=turn_id, domain=domain_name, slot=slot_name)
                    gt_value = unify_value(gt_value, subtask)
                    pred_value = unify_value(pred_domain[slot_name], subtask)
                    slot_tot += 1

                    if gt_value == pred_value or isinstance(gt_value, list) and pred_value in gt_value:
                        slot_acc += 1
                        if gt_value:
                            tp += 1
                    else:
                        errors.append([dialog_id, turn_id, domain_name, slot_name, gt_value, pred_value])
                        turn_result = False
                        if gt_value:
                            fn += 1
                        if pred_value:
                            fp += 1
            joint_acc += turn_result

    precision = tp / (tp + fp) if tp + fp else 1
    recall = tp / (tp + fn) if tp + fn else 1
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1
    return {
        'status': 'ok',
        'joint accuracy': joint_acc / joint_tot,
        'slot': {
            'accuracy': slot_acc / slot_tot,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    }, errors


def dump_result(model_dir, filename, result, errors=None, pred=None):
    output_dir = os.path.join('../results', model_dir)
    os.makedirs(output_dir, exist_ok=True)
    json.dump(result, open(os.path.join(output_dir, filename), 'w'), indent=4, ensure_ascii=False)
    if errors:
        import csv
        with open(os.path.join(output_dir, 'errors.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(errors)
    if pred:
        json.dump(pred, open(os.path.join(output_dir, 'pred.json'), 'w'), indent=4, ensure_ascii=False)
