import copy
import pandas as pd
import numpy as np
import argparse
import warnings
from model.evaluation import items
warnings.filterwarnings('ignore')


def unexpectedness(pi, pj, pij):
    if pij == 0.:
        return 1
    return -np.log(pij / (pi * pj)) / np.log(pij)


if __name__ == '__main__':
    """
    Compute serendipity, novelty, coverage
    USAGE: python utils/extra_metrics.py --model_version "final_model"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, required=True)
    args = parser.parse_known_args()[0]
    model_version = args.model_version

    it_probs = pd.read_csv('data/item_probs.csv', header=None)
    joint_probs = pd.read_csv('data/joint_probs.csv', header=None)
    it_probs = dict(zip(list(range(len(it_probs))), list(it_probs[it_probs.columns[1]])))
    joint_probs = dict(zip(list(joint_probs[joint_probs.columns[0]]), list(joint_probs[joint_probs.columns[1]])))
    for top_k in [1, 3, 5]:
        try:
            ud = pd.read_csv('data/{}_hist_pred_top_{}.csv'.format(model_version, top_k), header=None)
            ir = pd.read_csv('data/{}_item_recommended_top_{}.csv'.format(model_version, top_k), header=None)
        except:
            continue
        it_recs = dict(zip(list(ir[ir.columns[0]]), list(ir[ir.columns[1]])))
        it_recs_old = copy.deepcopy(it_recs)
        history = ud[ud.columns[0]]
        preds = ud[ud.columns[1]]
        real_preds = ud[ud.columns[2]]
        n_users = len(ud)
        cum_serendipity = []
        cum_novelty = []
        for i in range(n_users):
            pred = np.fromstring(preds[i][1:-1], dtype=int, sep=' ')
            real_pred = np.fromstring(real_preds[i][1:-1], dtype=int, sep=' ')
            hist = np.fromstring(history[i][1:-1], dtype=int, sep=' ')
            for p in pred:
                # novelty
                if p not in it_recs.keys():
                    it_recs[p] = 0
                nov = 1 - it_recs[p] / n_users
                cum_novelty.append(nov)
                # serendipity
                if p not in real_pred:
                    cum_serendipity.append(0)
                else:
                    unexpect = []
                    pi = it_probs[p]
                    for j in hist:
                        pj = it_probs[j]
                        if p == j:
                            pij = 1.
                            unexpect.append(0)
                        else:
                            pij = joint_probs[items[int(p)]+'-'+items[int(j)]]
                            unexpect.append(abs(unexpectedness(pi, pj, pij)))
                    if len(unexpect) == 0:
                        unexpect.append(0.)
                    cum_serendipity.append(np.mean(unexpect))
        print('k =', top_k)
        serendipity = np.mean(cum_serendipity)
        print('serendipity:', serendipity)
        novelty = np.mean(cum_novelty)
        print('novelty:', novelty)
        coverage = len(it_recs_old.keys()) / len(items)
        print('coverage', coverage)
