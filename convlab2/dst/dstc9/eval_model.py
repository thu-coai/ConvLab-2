"""
    evaluate DST model
"""

import os
import json
import importlib

from convlab2.dst import DST
from convlab2.dst.dstc9.utils import prepare_data, eval_states


def evaluate(model_name, subtask):
    subdir = 'multiwoz_zh' if subtask == 'multiwoz' else 'crosswoz_en'
    module = importlib.import_module(f'{model_name}.{subdir}')
    assert 'Model' in dir(module), 'please import your model as name `Model` in your subtask module root'
    model_cls = module.__getattribute__('Model')
    assert issubclass(model_cls, DST), 'the model must implement DST interface'
    # load weights, set eval() on default
    model = model_cls()
    gt = {}
    pred = {}
    for dialog_id, turns in prepare_data(subtask).items():
        gt_dialog = []
        pred_dialog = []
        model.init_session()
        for sys_utt, user_utt, gt_turn in turns:
            gt_dialog.append(gt_turn)
            pred_dialog.append(model.update_turn(sys_utt, user_utt))
        gt[dialog_id] = gt_dialog
        pred[dialog_id] = pred_dialog
    result = eval_states(gt, pred)
    print(result)
    json.dump(result, open(os.path.join(model_name, subdir, 'result.json'), 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('subtask', type=str, choices=['multiwoz', 'crosswoz'])
    args = parser.parse_args()
    evaluate('example', args.subtask)
