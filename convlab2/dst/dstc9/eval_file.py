"""
    evaluate output file
"""

from convlab2.dst.dstc9.utils import prepare_data, eval_states

if __name__ == '__main__':
    import os
    import json
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('subtask', type=str, choices=['multiwoz', 'crosswoz'])
    args = parser.parse_args()

    gt = {
        dialog_id: [state for _, _, state in turns]
        for dialog_id, turns in prepare_data(args.subtask).items()
    }
    # json.dump(gt, open('gt-crosswoz.json', 'w'), ensure_ascii=False, indent=4)

    results = {}
    for i in range(1, 6):
        filename = f'submission{i}.json'
        if not os.path.exists(filename):
            continue
        pred = json.load(open(filename))
        results[filename] = eval_states(gt, pred)

    json.dump(results, open('results.json', 'w'), indent=4, ensure_ascii=False)
