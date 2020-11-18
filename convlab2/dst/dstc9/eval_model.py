"""
    evaluate DST model
"""

import os
import json
import importlib

from tqdm import tqdm

from convlab2.dst import DST
from convlab2.dst.dstc9.utils import prepare_data, eval_states, dump_result, extract_gt


def evaluate(model_dir, subtask, test_data, gt):
    module = importlib.import_module(model_dir.replace('/', '.'))
    assert 'Model' in dir(module), 'please import your model as name `Model` in your subtask module root'
    model_cls = getattr(module, 'Model')
    assert issubclass(model_cls, DST), 'the model must implement DST interface'
    # load weights, set eval() on default
    model = model_cls()
    pred = {}
    bar = tqdm(total=sum(len(turns) for turns in test_data.values()), ncols=80, desc='evaluating')
    for dialog_id, turns in test_data.items():
        model.init_session()
        pred[dialog_id] = []
        for sys_utt, user_utt, gt_turn in turns:
            pred[dialog_id].append(model.update_turn(sys_utt, user_utt))
            bar.update()
    bar.close()

    result, errors = eval_states(gt, pred, subtask)
    print(json.dumps(result, indent=4))
    dump_result(model_dir, 'model-result.json', result, errors, pred)


def eval_team(team, correct_name_label):
    for subtask in ['multiwoz', 'crosswoz']:
        test_data = prepare_data(subtask, 'dstc9', correct_name_label=correct_name_label)
        gt = extract_gt(test_data)
        for i in range(1, 6):
            model_dir = os.path.join(team, f'{subtask}-dst', f'submission{i}')
            if not os.path.exists(model_dir):
                continue
            print(model_dir)
            evaluate(model_dir, subtask, test_data, gt)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--teams', type=str, nargs='*')
    parser.add_argument('correct_name_label', action='store_true')
    args = parser.parse_args()
    if not args.teams:
        for team in os.listdir('.'):
            if not os.path.isdir(team):
                continue
            eval_team(team, args.correct_name_label)
    else:
        for team in args.teams:
            eval_team(team, args.correct_name_label)
