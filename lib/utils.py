#!/usr/bin/python
# -*- coding: utf-8 -*-
from lib.kern_model import KERN
from lib.object_detector import Result
from lib.motifs_model import Motifs
from lib.imp_model import IMP
from lib.vctree_model import VCTree
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, eval_entry, calculate_mR_from_evaluator_list
from lib.pytorch_misc import optimistic_restore
from config import cfg, BOX_SCALE, IM_SCALE

from tqdm import tqdm
import torch
import pickle as pkl
import os
import numpy as np
import joblib


def define_model(cfg, ind_to_classes, ind_to_predicates):
    if cfg.model == 'kern':
        detector = KERN(classes=ind_to_classes, rel_classes=ind_to_predicates,
                        num_gpus=cfg.num_gpus, mode=cfg.mode, require_overlap_det=True,
                        use_resnet=cfg.use_resnet, use_proposals=cfg.use_proposals, pooling_dim=cfg.pooling_dim,
                        use_ggnn_obj=cfg.use_obj, ggnn_obj_time_step_num=cfg.obj_time_step_num,
                        ggnn_obj_hidden_dim=cfg.obj_hidden_dim, ggnn_obj_output_dim=cfg.obj_output_dim,
                        use_obj_knowledge=cfg.use_obj_knowledge, obj_knowledge=cfg.obj_knowledge,
                        use_ggnn_rel=True, ggnn_rel_time_step_num=cfg.ggnn_rel_time_step_num,
                        ggnn_rel_hidden_dim=cfg.ggnn_rel_hidden_dim, ggnn_rel_output_dim=cfg.ggnn_rel_output_dim,
                        use_rel_knowledge=True, rel_knowledge=cfg.rel_knowledge)
    elif cfg.model == 'motifs':
        detector = Motifs(classes=ind_to_classes, rel_classes=ind_to_predicates,
                          num_gpus=cfg.num_gpus, mode=cfg.mode, require_overlap_det=True,
                          use_resnet=cfg.use_resnet, order=cfg.order,
                          nl_edge=cfg.nl_edge, nl_obj=cfg.nl_obj, hidden_dim=cfg.motifs_hidden_dim,
                          use_proposals=cfg.use_proposals,
                          pass_in_obj_feats_to_decoder=cfg.pass_in_obj_feats_to_decoder,
                          pass_in_obj_feats_to_edge=cfg.pass_in_obj_feats_to_edge,
                          pooling_dim=cfg.pooling_dim,
                          rec_dropout=cfg.rec_dropout,
                          use_bias=True,
                          use_tanh=False,
                          limit_vision=False
                          )
    elif cfg.model == 'imp':
        detector = IMP(classes=ind_to_classes, rel_classes=ind_to_predicates,
                          num_gpus=cfg.num_gpus, mode=cfg.mode, require_overlap_det=True,
                          use_resnet=cfg.use_resnet, order=cfg.order,
                          nl_edge=cfg.nl_edge, nl_obj=cfg.nl_obj, hidden_dim=256,
                          use_proposals=cfg.use_proposals,
                          pass_in_obj_feats_to_decoder=cfg.pass_in_obj_feats_to_decoder,
                          pass_in_obj_feats_to_edge=cfg.pass_in_obj_feats_to_edge,
                          pooling_dim=cfg.pooling_dim,
                          rec_dropout=cfg.rec_dropout,
                          use_bias=True,
                          use_tanh=False,
                          limit_vision=False
                          )
    elif cfg.model == 'vctree':
        detector = VCTree(classes=ind_to_classes, rel_classes=ind_to_predicates,
                            num_gpus=cfg.num_gpus, mode=cfg.mode, require_overlap_det=True,
                            use_resnet=cfg.use_resnet, order='confidence',
                            nl_edge=1, nl_obj=1, hidden_dim=512,
                            pass_in_obj_feats_to_decoder=cfg.pass_in_obj_feats_to_decoder,
                            pass_in_obj_feats_to_edge=cfg.pass_in_obj_feats_to_edge,
                            use_proposals=cfg.use_proposals,
                            pooling_dim=cfg.pooling_dim,
                            rec_dropout=0.0,
                            use_bias=True,
                            use_tanh=False,
                            use_encoded_box=True,
                            use_rl_tree=cfg.use_rl_tree,
                            limit_vision=False
                            )
    else:
        raise ValueError('Unexpected model type')
    return detector


def load_ckpt(detector, ckpt_path):
    ckpt = torch.load(ckpt_path)
    if 'faster-rcnn' not in os.path.basename(ckpt_path):
        print("Loading EVERYTHING")

        start_epoch = -1
        optimizer, scheduler = None, None
        if optimistic_restore(detector, ckpt['state_dict']) and cfg.resume_training:
            start_epoch = ckpt['epoch']
            optimizer = ckpt['optimizer'] if hasattr(ckpt, 'optimizer') else None
            scheduler = ckpt['scheduler'] if hasattr(ckpt, 'scheduler') else None
    else:
        start_epoch = -1
        optimizer, scheduler = None, None
        if hasattr(detector, 'detector'):
            optimistic_restore(detector.detector, ckpt['state_dict'])

        detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

        detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    return start_epoch, optimizer, scheduler


def print_eval_stat(eval_list, name):
    eval_list = [x*100 for x in eval_list]  # Convert to percentage
    print('%s\tAvg: %.2f\tMin: %.2f\tMax: %.2f\tStd: %.2f' %
          (name, np.mean(eval_list), np.min(eval_list), np.max(eval_list), np.std(eval_list)))


def do_test(detector, test, test_loader):

    def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
        det_res = detector[b]

        if cfg.cache_det_res:
            obj_det_res = det_res
            keys_to_cache = ['im_inds', 'rm_box_priors', 'rm_obj_dists', 'fmap', 'boxes_all', 'rm_obj_labels', 'rel_labels']
            res_to_cache = Result()
            for k, v in obj_det_res.__dict__.items():
                if k in keys_to_cache and v is not None:
                    if isinstance(v, torch.Tensor):
                        v = v.cpu()
                    setattr(res_to_cache, k, v.cpu())
            # Save results for each batch independently
            img_id = b.img_ids[0]
            cache_path = os.path.join(os.path.dirname(cfg.cache), 'obj_det-%s' % cfg.test_data_name, '%s.pkl' % img_id)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pkl.dump(res_to_cache, f)
            return

        if cfg.cache_obj_dists:
            det_res, obj_dists = det_res
            all_obj_dists.append(obj_dists.data.cpu().numpy())

        if cfg.cache_gaussians:
            pred = det_res[-1].argmax(1)
            all_gaussians.append(
                (detector.mu.data.cpu(), detector.log_var.data.cpu(), pred)
            )
            return

        if cfg.num_gpus == 1:
            det_res = [det_res]

        for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
            gt_entry = {
                'gt_classes': test.gt_classes[batch_num + i].copy(),
                'gt_relations': test.relationships[batch_num + i].copy(),
                'gt_boxes': test.gt_boxes[batch_num + i].copy(),
            }
            assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
            # assert np.all(rels_i[:,2] > 0)

            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,
            }
            all_pred_entries.append(pred_entry)

            eval_entry(cfg.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                       evaluator_list, evaluator_multiple_preds_list)

    all_pred_entries = []
    evaluator = BasicSceneGraphEvaluator.all_modes()
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    if cfg.cache_obj_dists:
        all_obj_dists = []
    if cfg.cache_gaussians:
        all_gaussians = []
    ind_to_predicates = test.ind_to_predicates  # ind_to_predicates[0] means no relationship
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

    if cfg.cache is not None and os.path.exists(cfg.cache):
        print("Found {}! Loading from it".format(cfg.cache))
        with open(cfg.cache, 'rb') as f:
            cache = pkl.load(f)
        if cfg.use_pred_entries_cache or isinstance(cache, list):
            if isinstance(cache, list):  # to be compatible with older version
                all_pred_entries = cache
            else:
                all_pred_entries = cache['pred_entries']
            for i, pred_entry in enumerate(tqdm(all_pred_entries)):
                gt_entry = {
                    'gt_classes': test.gt_classes[i].copy(),
                    'gt_relations': test.relationships[i].copy(),
                    'gt_boxes': test.gt_boxes[i].copy(),
                }

                eval_entry(cfg.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                           evaluator_list, evaluator_multiple_preds_list)

            # calculate_mR_from_evaluator_list(evaluator_list, cfg.mode)

            recall, mean_recall, prd_recall, _ = evaluator[cfg.mode].print_stats()
            eval_results = {
                'recall': recall,
                'mean_recall': mean_recall,
                'prd_recall': prd_recall
            }
            result = {
                'pred_entries': all_pred_entries,
                'eval_results': eval_results
            }
            with open(cfg.cache, 'wb') as f:
                pkl.dump(result, f)
        else:
            eval_results = cache['eval_results']
            for k, v in eval_results['recall'].items():
                print('%s: %.2f' % (k, v * 100))
            for k, v in eval_results['mean_recall'].items():
                print('%s: %.2f' % (k, v * 100))
    else:
        if cfg.test_data_name == 'train':
            detector.train()
        else:
            detector.eval()
        if cfg.visual_gaussian:
            inference_times = cfg.inference_times
        else:
            inference_times = 1
        recall_list = []
        mean_recall_list = []
        for i in range(inference_times):
            all_pred_entries = []
            with torch.no_grad():
                for val_b, batch in enumerate(tqdm(test_loader)):
                    val_batch(cfg.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                          evaluator_multiple_preds_list)

            if cfg.cache_det_res:
                return

            # calculate_mR_from_evaluator_list(evaluator_list, cfg.mode)
            recall, mean_recall, prd_recall, _ = evaluator[cfg.mode].print_stats()
            recall_list.append(recall)
            mean_recall_list.append(mean_recall)
            eval_results = {
                'recall': recall,
                'mean_recall': mean_recall,
                'prd_recall': prd_recall
            }
            result = {
                'pred_entries': all_pred_entries,
                'eval_results': eval_results
            }

            if cfg.cache is not None:
                if cfg.cache_obj_dists:
                    cache_path = os.path.join(os.path.dirname(cfg.cache), 'obj_dists-%s.pkl' % cfg.test_data_name)
                    with open(cache_path, 'wb') as f:
                        pkl.dump(all_obj_dists, f)
                        print('obj dists cache saved at %s' % cache_path)
                    exit()

                if cfg.cache_gaussians:
                    cache_path = os.path.join(os.path.dirname(cfg.cache), 'gaussians-%s.jbl' % cfg.test_data_name)
                    joblib.dump(all_gaussians, cache_path)
                    print('gaussians cache saved at %s' % cache_path)
                    exit()

                if inference_times > 1:
                    cache_dir = os.path.dirname(cfg.cache)
                    cache_name = os.path.basename(cfg.cache)
                    splited = cache_name.split('.')
                    cache_path = os.path.join(cache_dir, '%s-%d.%s' % (splited[0], i, splited[1]))
                else:
                    cache_path = cfg.cache
                with open(cache_path, 'wb') as f:
                    pkl.dump(result, f)

        if inference_times > 1:
            for k in recall.keys():
                print_eval_stat([x[k] for x in recall_list], k)
            for k in mean_recall.keys():
                print_eval_stat([x[k] for x in mean_recall_list], k)
