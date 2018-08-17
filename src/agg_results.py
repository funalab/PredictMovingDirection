import os
import json
import argparse
import collections as cl
import numpy as np

FOLDS = 4
keys = ["validation/main/accuracy", "validation_in_mca/main/mca"]

parser = argparse.ArgumentParser(description='Run guided grad-gcam')
parser.add_argument('--results', default='./results',
                    help='Cross validation results path')
args = parser.parse_args()

res = cl.defaultdict(list)
for n in range(FOLDS):
    bs_path = os.path.join(args.results, 'fold{}'.format(n), 'best_score.json')
    with open(bs_path, 'r') as fp:
        bs = json.load(fp)
    for k in keys:
        res[k].append(bs[k])

summary = cl.OrderedDict()
for k in keys:
    values = np.array(res[k])
    summary["mean_"+k] = np.mean(values)
    summary["sd_"+k] = np.std(values)

with open(os.path.join(args.results, 'summary.json'), 'w') as fp:
    json.dump(summary, fp)
