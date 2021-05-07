from glob import glob
import pickle as pkl
import copy
import math
import os
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from config import cfg

if __name__ == '__main__':
    preds_list = []
    all_paths = list(glob(cfg.cache + '*.pkl'))
    for path in all_paths:
        if path.endswith('ensemble.pkl'):
            continue
        print('Loading cache from %s' % path)
        with open(path, 'rb') as f:
            cache = pkl.load(f)
        preds_list.append(cache['pred_entries'])

    res = copy.deepcopy(preds_list[0])
    for entry_id, res_entry in tqdm(enumerate(res)):
        res_entry['rel_scores'] = []

        for i in range(len(preds_list[0][entry_id]['rel_scores'])):
            # pick by voting
            pred2scores_row = defaultdict(list)
            for j in range(len(preds_list)):
                rel_scores_row = preds_list[j][entry_id]['rel_scores'][i]
                assert rel_scores_row.shape == (51,)
                pred = rel_scores_row.argmax()
                pred2scores_row[pred].append(rel_scores_row)
            max_votes = 0
            final_scores_row = None
            for k, v in pred2scores_row.items():
                votes = len(v)
                if votes > max_votes:
                    max_votes = votes
                    final_scores_row = random.choice(v)
            res_entry['rel_scores'].append(final_scores_row)

        res_entry['rel_scores'] = np.stack(res_entry['rel_scores'])
        # Sort by scores
        obj_scores = res_entry['obj_scores']
        rel_inds = res_entry['pred_rel_inds']
        pred_scores = res_entry['rel_scores']
        obj_scores0 = obj_scores[rel_inds[:, 0]]
        obj_scores1 = obj_scores[rel_inds[:, 1]]
        pred_scores_max = pred_scores[:, 1:].max(1)

        rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
        rel_scores_vs = np.sort(rel_scores_argmaxed)[::-1]
        rel_scores_idx = np.argsort(rel_scores_argmaxed)[::-1]

        res_entry['pred_rel_inds'] = rel_inds[rel_scores_idx]
        res_entry['rel_scores'] = pred_scores[rel_scores_idx]
        assert res_entry['rel_scores'].shape == preds_list[0][entry_id]['rel_scores'].shape

    out_path = cfg.cache + 'ensemble.pkl'
    with open(out_path, 'wb') as f:
        pkl.dump(res, f)
        print('Results saved at %s' % out_path)
