#!/usr/bin/python
# -*- coding: utf-8 -*-
from lib.utils import define_model, load_ckpt
from dataloaders.visual_genome import VG, vg_collate
from lib.bimodal import gaussian_entropy
from config import cfg, OLD_DATA_PATH
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
import os
import json
import pickle
import pandas as pd

test_data = VG('test', num_val_im=cfg.val_size, filter_duplicate_rels=True,
               use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
               num_im=cfg.num_im)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=cfg.num_gpus,
    shuffle=False,
    num_workers=cfg.num_workers,
    collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=cfg.num_gpus, is_train=False),
    drop_last=True,
    pin_memory=True,
)

detector = define_model(cfg, test_data.ind_to_classes, test_data.ind_to_predicates)
load_ckpt(detector, cfg.ckpt)
detector.cuda()
detector.eval()

prd_label_list = ['no_relation']
with open(os.path.join(OLD_DATA_PATH, 'vg', 'predicates.json')) as f:
    prd_label_list += json.load(f)  # a list of labels
obj_label_list = ['background']
with open(os.path.join(OLD_DATA_PATH, 'vg', 'objects.json')) as f:
    obj_label_list += json.load(f)  # a list of labels

# Measure uncertainty class-separately for both visual embeddings and predicate embeddings
vis_uncertainty = []
word_uncertainty = []  # There's only one batch for words
num_class = 51
assert (cfg.use_bimodal_rel and cfg.use_gaussian) or cfg.visual_gaussian
for val_b, batch in enumerate(tqdm(test_loader)):
    with torch.no_grad():
        detector[batch]

    if cfg.use_bimodal_rel:
        for i in range(num_class):
            prd_vis_embed = detector.rel_compress.prd_vis_embed
            cur_class_prd_vis_embed = prd_vis_embed.view([prd_vis_embed.shape[0], num_class, -1])[:, i]
            vis_uncertainty_i = float(gaussian_entropy(cur_class_prd_vis_embed))
            if val_b == 0:
                # Only need to assign once for word
                cur_class_prd_word_embed = detector.rel_compress.prd_word_embed[i].unsqueeze(0)
                word_uncertainty.append(float(gaussian_entropy(cur_class_prd_word_embed)))
                vis_uncertainty.append([vis_uncertainty_i])
            else:
                vis_uncertainty[i].append(vis_uncertainty_i)
    elif cfg.visual_gaussian:
        vis_uncertainty_i = torch.mean(detector.log_var, 1).data.cpu().numpy()
        vis_uncertainty.append(vis_uncertainty_i)


def cal_mean(data_list):
    return 1.0 * sum(data_list) / len(data_list)


if cfg.use_bimodal_rel:
    for i in range(num_class):
        vis_uncertainty[i] = cal_mean(vis_uncertainty[i])

    df = pd.DataFrame({
        'predicate': prd_label_list + ['MEAN'],
        'vis_uncertainty': vis_uncertainty + [cal_mean(vis_uncertainty)],
        'word_uncertainty': word_uncertainty + [cal_mean(word_uncertainty)],
    })

    print(df)
else:
    cache_path = os.path.join(os.path.dirname(cfg.ckpt), 'caches/uncertainty.pkl')
    pickle.dump(vis_uncertainty, open(cache_path, 'wb'))
    print('cache saved at', cache_path)
