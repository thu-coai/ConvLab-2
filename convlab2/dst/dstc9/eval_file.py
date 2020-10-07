"""
    evaluate output file
"""

import json
import os

from convlab2.dst.dstc9.utils import prepare_data, extract_gt, eval_states, get_subdir


def evaluate(model_dir, subtask, gt):
    subdir = get_subdir(subtask)
    results = {}
    for i in range(1, 6):
        filepath = os.path.join(model_dir, subdir, f'submission{i}.json')
        if not os.path.exists(filepath):
            continue
        pred = json.load(open(filepath))
        results[i] = eval_states(gt, pred, subtask)

    print(json.dumps(results, indent=4))
    json.dump(results, open(os.path.join(model_dir, subdir, 'file-results.json'), 'w'), indent=4, ensure_ascii=False)


# generate submission examples
def dump_example(subtask, split):
    test_data = prepare_data(subtask, split)
    pred = extract_gt(test_data)
    json.dump(pred, open(os.path.join('example', get_subdir(subtask), 'submission1.json'), 'w'), ensure_ascii=False, indent=4)
    import random
    for dialog_id, states in pred.items():
        for state in states:
            for domain in state.values():
                for slot, value in domain.items():
                    if value:
                        if random.randint(0, 2) == 0:
                            domain[slot] = ""
                    else:
                        if random.randint(0, 4) == 0:
                            domain[slot] = "2333"
    json.dump(pred, open(os.path.join('example', get_subdir(subtask), 'submission2.json'), 'w'), ensure_ascii=False, indent=4)
    for dialog_id, states in pred.items():
        for state in states:
            for domain in state.values():
                for slot in domain:
                    domain[slot] = ""
    json.dump(pred, open(os.path.join('example', get_subdir(subtask), 'submission3.json'), 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('subtask', type=str, choices=['multiwoz', 'crosswoz'])
    parser.add_argument('split', type=str, choices=['train', 'val', 'test', 'human_val'])
    args = parser.parse_args()
    subtask = args.subtask
    split = args.split
    dump_example(subtask, split)
    test_data = prepare_data(subtask, split)
    gt = extract_gt(test_data)
    evaluate('example', subtask, gt)
