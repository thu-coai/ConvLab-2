"""
    evaluate DST model
"""

import os
import json
import importlib

from convlab2.dst import DST
from convlab2.dst.dstc9.utils import prepare_data, eval_states, get_subdir


def evaluate(model_dir, subtask, test_data, gt):
    subdir = get_subdir(subtask)
    module = importlib.import_module(f'{model_dir}.{subdir}')
    assert 'Model' in dir(module), 'please import your model as name `Model` in your subtask module root'
    model_cls = module.__getattribute__('Model')
    assert issubclass(model_cls, DST), 'the model must implement DST interface'
    # load weights, set eval() on default
    model = model_cls()
    pred = {}
    for dialog_id, turns in test_data.items():
        model.init_session()
        pred[dialog_id] = [model.update_turn(sys_utt, user_utt) for sys_utt, user_utt, gt_turn in turns]
    result = eval_states(gt, pred, subtask)
    print(json.dumps(result, indent=4))
    json.dump(result, open(os.path.join(model_dir, subdir, 'model-result.json'), 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('subtask', type=str, choices=['multiwoz', 'crosswoz'])
    parser.add_argument('split', type=str, choices=['train', 'val', 'test', 'human_val'])
    args = parser.parse_args()
    subtask = args.subtask
    test_data = prepare_data(subtask, args.split)
    gt = {
        dialog_id: [state for _, _, state in turns]
        for dialog_id, turns in test_data.items()
    }
    evaluate('example', subtask, test_data, gt)
