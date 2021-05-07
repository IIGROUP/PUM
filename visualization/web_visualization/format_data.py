#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Example:
python visualization/web_visualization/format_data.py \
-cache checkpoints/sgdet-baseline-sgcls_init-lr1e-6/caches/test_prediction-nms_thresh0.5.pkl \
-cache2 checkpoints/sgdet-motifs-sgcls_init/caches/test_prediction.pkl -nimg 1000 \
-m sgdet
"""
import numpy as np
from tqdm import tqdm
import json
import orjson
import pickle
import time
import os
from collections import defaultdict
from config import cfg, OLD_DATA_PATH
from dataloaders.visual_genome import VG


def orjson_dump(file_path, data):
    with open(file_path, 'w') as f:
        f.write(
            str(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY), 'utf-8'))


def load_pred_entries(path):
    with open(path, 'rb') as f:
        cache = pickle.load(f)
        if isinstance(cache, list):
            all_pred_entries = cache
        else:
            all_pred_entries = cache['pred_entries']
    if cfg.num_im != -1:
        all_pred_entries = all_pred_entries[:cfg.num_im]
    return all_pred_entries


def add_pred_data(entry, pred_entry, gt_rels, uncertainty=None):
    mode = cfg.mode
    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
    assert pred_rels.shape[1] == 3, 'should be the same format as gt_rels, shape of (X, 3)'
    entry[mode + '_pred_rels'] = pred_rels
    entry[mode + '_pred_rels_scores'] = rel_scores[:, 1:].max(1)

    entry[mode + '_pred_scores'] = pred_entry['obj_scores']
    entry[mode + '_pred_classes'] = pred_entry['pred_classes']

    entry['pred_boxes'] = pred_entry['pred_boxes']

    if uncertainty is not None:
        entry['uncertainty'] = uncertainty

    # add [mode]_gt2pred_rel and [mode]_pred2gt_rel
    entry[mode + '_gt2pred_rel'] = [list() for _ in range(len(gt_rels))]
    entry[mode + '_pred2gt_rel'] = [list() for _ in range(len(pred_rels))]
    pred_rels_dict = dict()
    for ind, rel in enumerate(pred_rels[:100]):  # only consider the first 100 entries
        pred_rels_dict[tuple(rel)] = ind
    for gt_ind, gt_rel in enumerate(gt_rels):
        rel = tuple(gt_rel)
        if rel in pred_rels_dict.keys():
            pred_ind = pred_rels_dict[rel]
            entry[mode + '_gt2pred_rel'][gt_ind].append(pred_ind)
            entry[mode + '_pred2gt_rel'][pred_ind].append(gt_ind)


def main():
    test_data = VG('test', num_val_im=cfg.val_size, filter_duplicate_rels=True,
                   use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
                   num_im=cfg.num_im)

    # Get names.json
    result_dir = 'visualization/web_visualization'
    with open(os.path.join(OLD_DATA_PATH, 'vg', 'predicates.json')) as f:
        prd_label_list = json.load(f)  # a list of labels
    with open(os.path.join(OLD_DATA_PATH, 'vg', 'objects.json')) as f:
        obj_label_list = json.load(f)  # a list of labels
    names = {}
    bg = ['__background__']
    names['preds'] = bg + prd_label_list
    names['classes'] = bg + obj_label_list
    json.dump(names, open(os.path.join(result_dir, 'names.json'), 'w'))

    # Get viz_results.json
    # Each entry should contain keys:
    # fn,
    # gt_classes, gt_box_scores, gt_boxes, gt_rels, pred_boxes,
    # [mode]_gt2pred_box, [mode]_pred2gt_box, [mode]_gt2pred_rel, [mode]_pred2gt_rel,
    # [mode]_pred_rels, [mode]_pred_classes, [mode]_pred_scores,
    # [mode]_gt_rels_scores, [mode]_gt_rels_scores_cls, [mode]_pred_rels_scores, [mode]_pred_rels_scores_cls
    # mode = pred_cls/sg_cls/sg_det
    viz_results = []
    key_word2im_ind = defaultdict(list)  # record image indexes for each predicate and object label
    all_pred_entries = load_pred_entries(cfg.cache)
    all_pred_entries_baseline = load_pred_entries(cfg.cache2)
    uncertainty_path = os.path.join(os.path.dirname(cfg.cache), 'uncertainty.pkl')
    if os.path.exists(uncertainty_path):
        uncertainty_records = pickle.load(open(uncertainty_path, 'rb'))
    else:
        uncertainty_records = None

    for i, (pred_entry, pred_entry_baseline) in enumerate(tqdm(zip(all_pred_entries, all_pred_entries_baseline))):
        entry = dict()

        entry['gt_classes'] = test_data.gt_classes[i]
        entry['gt_boxes'] = test_data.gt_boxes[i]
        entry['gt_rels'] = test_data.relationships[i]

        for rel in entry['gt_rels']:
            pred_label = names['preds'][rel[2]]
            sbj_gt_class = entry['gt_classes'][rel[0]]
            obj_gt_class = entry['gt_classes'][rel[1]]
            sbj_label = names['classes'][sbj_gt_class]
            obj_label = names['classes'][obj_gt_class]
            key_word2im_ind[pred_label].append(i)
            key_word2im_ind[sbj_label].append(i)
            key_word2im_ind[obj_label].append(i)

        entry['fn'] = test_data.filenames[i]

        entry['pred'] = {'gt_boxes': entry['gt_boxes']}
        entry['pred_baseline'] = {'gt_boxes': entry['gt_boxes']}

        if uncertainty_records is not None:
            uncertainty_i = uncertainty_records[i]
        else:
            uncertainty_i = None

        add_pred_data(entry['pred'], pred_entry, entry['gt_rels'], uncertainty_i)
        add_pred_data(entry['pred_baseline'], pred_entry_baseline, entry['gt_rels'])

        entry[cfg.mode + '_gt2pred_rel'] = entry['pred'][cfg.mode + '_gt2pred_rel']

        viz_results.append(entry)

    orjson_dump(os.path.join(result_dir, 'viz_results.json'), viz_results)
    orjson_dump(os.path.join(result_dir, 'key_word2im_ind.json'), key_word2im_ind)


if __name__ == '__main__':
    main()
