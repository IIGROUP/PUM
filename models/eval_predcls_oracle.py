import os
from glob import glob
from tqdm import tqdm
import pickle as pkl
import numpy as np
from functools import reduce
from collections import defaultdict
from dataloaders.visual_genome import VG, vg_collate
from torch.utils.data import DataLoader
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, evaluate_recall
from config import cfg

if __name__ == '__main__':
    preds_list = []
    if os.path.isdir(cfg.cache):
        num_inferences = cfg.inference_times
        all_paths = [os.path.join(cfg.cache, 'test_prediction-%d.pkl' % i) for i in range(num_inferences)]
    else:
        all_paths = [cfg.cache]
    for path in all_paths:
        if path.endswith('ensemble.pkl'):
            continue
        print('Loading cache from %s' % path)
        with open(path, 'rb') as f:
            cache = pkl.load(f)
        preds_list.append(cache['pred_entries'])

    test_data = VG(cfg.test_data_name, num_val_im=5000, filter_duplicate_rels=True,
                   use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
                   num_im=cfg.num_im)

    recall = []
    for i, gt_entry in tqdm(enumerate(test_data)):
        # Compute oracle recall
        gt_rels = gt_entry['gt_relations']
        gt_boxes = gt_entry['gt_boxes'].astype(float)
        gt_classes = gt_entry['gt_classes']

        multiple_match_union = []
        for preds in preds_list:
            pred_entry = preds[i]
            pred_boxes = pred_entry['pred_boxes'].astype(float)
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']
            pred_rel_inds = pred_entry['pred_rel_inds']
            rel_scores = pred_entry['rel_scores']
            pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
            predicate_scores = rel_scores[:,1:].max(1)
            pred_to_gt, _, _ = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores)
            match = reduce(np.union1d, pred_to_gt)  # no constraint for top-k confidences
            multiple_match_union.append(match)
        if multiple_match_union:
            multiple_match_union = reduce(np.union1d, multiple_match_union)
        rec_i = float(len(multiple_match_union)) / float(gt_rels.shape[0])
        recall.append(rec_i)
    print('oracle recall: %f' % float(np.mean(recall)))
