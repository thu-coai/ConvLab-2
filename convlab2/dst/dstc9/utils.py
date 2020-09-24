import os
import json
import zipfile

from convlab2 import DATA_ROOT


def get_subdir(subtask):
    subdir = 'multiwoz_zh' if subtask == 'multiwoz' else 'crosswoz_en'
    return subdir


def prepare_data(subtask, split, data_root=DATA_ROOT):
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
                state = {}
                for domain_name, domain in turns[i + 1]['metadata'].items():
                    if domain_name in ['警察机关', '医院']:
                        continue
                    domain_state = {}
                    for slots in domain.values():
                        for slot_name, value in slots.items():
                            domain_state[slot_name] = value
                    state[domain_name] = domain_state
                dialog_data.append((sys_utt, user_utt, state))
            data[dialog_id] = dialog_data
    else:
        for dialog_id, dialog in test_data.items():
            dialog_data = []
            turns = dialog['messages']
            for i in range(0, len(turns), 2):
                sys_utt = turns[i - 1]['content'] if i else None
                user_utt = turns[i]['content']
                state = {}
                for domain_name, domain in turns[i + 1]['sys_state_init'].items():
                    domain_state = {}
                    for slot_name, value in domain.items():
                        if slot_name == 'selectedResults':
                            continue
                        domain_state[slot_name] = value
                    state[domain_name] = domain_state
                dialog_data.append((sys_utt, user_utt, state))
            data[dialog_id] = dialog_data

    return data


def extract_gt(test_data):
    gt = {
        dialog_id: [state for _, _, state in turns]
        for dialog_id, turns in test_data.items()
    }
    return gt


def eval_states(gt, pred, subtask):
    # for unifying values with the same meaning to the same expression
    value_unifier = {
        'multiwoz': {

        },
        'crosswoz': {
            '未提及': '',
        }
    }[subtask]

    def unify_value(value):
        return value_unifier.get(value, value)

    def exception(description, **kargs):
        ret = {
            'status': 'exception',
            'description': description,
        }
        for k, v in kargs.items():
            ret[k] = v
        return ret

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
                    gt_value = unify_value(gt_value)
                    pred_value = unify_value(pred_domain[slot_name])
                    slot_tot += 1
                    if gt_value == pred_value:
                        slot_acc += 1
                        if gt_value:
                            tp += 1
                    else:
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
        # 'slot accuracy': slot_acc / slot_tot,
        'slot': {
            'accuracy': slot_acc / slot_tot,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    }
