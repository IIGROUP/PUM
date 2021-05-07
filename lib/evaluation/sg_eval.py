"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
import math
import pickle
from collections import defaultdict
from functools import reduce
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import MODES
np.set_printoptions(precision=3)

class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_recall_hit'] = {}
        self.result_dict[mode + '_recall_count'] = {}
        for k in self.result_dict[mode + '_recall']:
            self.result_dict[mode + '_recall_hit'][k] = [0] * 51
            self.result_dict[mode + '_recall_count'][k] = [0] * 51

        self.multiple_preds = multiple_preds

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        if self.multiple_preds:
            recall_method = 'recall without constraint'
        else:
            recall_method = 'recall with constraint'
        recall = {}
        mean_recall = {}
        prd_recall = {}
        print('======================' + self.mode + '  ' + recall_method + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            recall_k = float(np.mean(v))
            print('R@%i: %.2f' % (k, recall_k * 100))
            recall['R@%i' % k] = recall_k

            prd_recall[k] = {}
            avg = 0.0
            for idx in range(1, 51):
                # The difference from the default mean recall computing
                # in that evaluate per batch or as a whole val set
                # tmp_avg = float(self.result_dict[self.mode + '_recall_hit'][k][idx+1]) / float(self.result_dict[self.mode + '_recall_count'][k][idx+1] + 1e-10)
                count = self.result_dict[self.mode + '_recall_count'][k][idx]
                if count == 0:
                    prd_recall[k][idx] = 0.0
                    continue
                hit = self.result_dict[self.mode + '_recall_hit'][k][idx]
                tmp_recall = float(hit) / float(count)
                avg += tmp_recall
                prd_recall[k][idx] = tmp_recall
            mean_recall_k = avg / 50.0
            print('MR@%i: %.2f' % (k, mean_recall_k * 100))
            mean_recall['mR@%i' % k] = mean_recall_k
        acc = None
        if 'obj_correct' in self.result_dict.keys():
            acc = self.result_dict['obj_correct'] / self.result_dict['obj_total']
            print('ObjCls Acc: %f' % acc)
        if 'ap' in self.result_dict.keys():
            ap = sum(self.result_dict['ap']) / len(self.result_dict['ap'])
            prec = sum(self.result_dict['prec']) / len(self.result_dict['prec'])
            rec = sum(self.result_dict['rec']) / len(self.result_dict['rec'])
            print('AP: %f' % ap)
            print('Precision: %f' % prec)
            print('Recall: %f' % rec)
        if 'tp' in self.result_dict.keys():
            mean_ap = []
            for cls in self.result_dict['gt_cls_counts'].keys():
                prec, rec = [], []
                tp_cum_sum, fp_cum_sum = 0, 0
                for i in range(len(self.result_dict['tp'][cls])):
                    tp = self.result_dict['tp'][cls][i] + tp_cum_sum
                    fp = self.result_dict['fp'][cls][i] + fp_cum_sum
                    tp_cum_sum = tp
                    fp_cum_sum = fp
                    prec.append(tp / (tp + fp))
                    rec.append(tp / self.result_dict['gt_cls_counts'][cls])
                ap = voc_ap(rec, prec)
                # print('[Class %-3d] AP: %f' % (cls, ap))
                mean_ap.append(ap)
            mean_ap = sum(mean_ap) / len(mean_ap)
            print('mAP: %f' % mean_ap)
        return recall, mean_recall, prd_recall, acc


def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                       viz_dict=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict: 
    :param viz_dict: 
    :param kwargs: 
    :return: 
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    pred_ = pred_entry['rel_scores'].copy()
    if mode in ['predcls', 'sgcls', 'sgdet', 'phrdet', 'objcls', 'objdet']:
        # Some pred_x has already assigned to GT
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError('invalid mode')

    if mode in ['sgcls', 'objcls']:
        # Evaluate accuracy of object classification
        if 'obj_correct' not in result_dict.keys():
            result_dict['obj_correct'] = 0
            result_dict['obj_total'] = 0
        result_dict['obj_correct'] += np.sum(pred_classes == gt_classes)
        result_dict['obj_total'] += len(gt_classes)

    if mode in ['sgdet', 'objdet']:
        iou_thres = 0.5
        # Evaluate mAP by counting FP and TP for each class
        if 'tp' not in result_dict.keys():
            result_dict['tp'] = defaultdict(list)
            result_dict['fp'] = defaultdict(list)
            result_dict['gt_cls_counts'] = defaultdict(int)
        for cls in gt_classes:
            result_dict['gt_cls_counts'][cls] += 1

        # Evaluate AP
        if 'ap' not in result_dict.keys():
            result_dict['ap'] = []
            result_dict['prec'] = []
            result_dict['rec'] = []
        # Rank the detections in descending order
        sorted_idx = np.argsort(obj_scores)[::-1]
        sorted_pred_classes = pred_classes[sorted_idx]
        sorted_pred_boxes = pred_boxes[sorted_idx]
        iou = bbox_overlaps(sorted_pred_boxes, gt_boxes)
        prec, rec = [], []
        tp = 0
        for idx, pred_cls in enumerate(sorted_pred_classes):
            cand_idx = (iou[idx] >= iou_thres)
            hit = False
            if len(np.where(gt_classes[cand_idx] == pred_cls)[0]):
                # Make a hit
                hit = True
                tp += 1
            result_dict['tp'][pred_cls].append(1 if hit else 0)
            result_dict['fp'][pred_cls].append(0 if hit else 1)
            prec.append(
                1.0 * tp / (idx + 1)
            )
            rec.append(
                1.0 * tp / len(gt_classes)
            )
        ap = voc_ap(rec, prec)
        result_dict['ap'].append(ap)
        result_dict['prec'].append(tp / len(pred_classes))  # overall precision
        result_dict['rec'].append(tp / len(gt_classes))  # overall precision

    if multiple_preds:
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:,0]], score_inds[:,1]+1))
        predicate_scores = rel_scores[score_inds[:,0], score_inds[:,1]+1]
    else:
        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        predicate_scores = rel_scores[:,1:].max(1)

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])

        # Compute hit and count for mean recall
        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]),2]
            result_dict[mode + '_recall_hit'][k][int(local_label)] += 1
            result_dict[mode + '_recall_hit'][k][0] += 1

        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx,2]
            result_dict[mode + '_recall_count'][k][int(local_label)] += 1
            result_dict[mode + '_recall_count'][k][0] += 1



        rec_i = float(len(match)) / float(gt_rels.shape[0])
        
        result_dict[mode + '_recall'][k].append(rec_i)


    return pred_to_gt, pred_5ples, rel_scores



###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]

    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    #assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    scores_overall = relation_scores.prod(1)
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")
    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )
    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt







def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False, save_file=None):
    # old codes from KERN, which make mistake to compute mean recall per batch
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        rel_results = {}
        for k, v in evaluator_rel[mode].result_dict[mode + '_recall'].items():
            rel_results['R@%i' % k] = np.mean(v)
        all_rel_results[pred_name] = rel_results
    
    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall

    print('\n')
    print('======================Wrong mean recall from KERN============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)

    return mean_recall





def eval_entry(mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    evaluator_multiple_preds[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    for (pred_id, _, evaluator_rel), (_, _, evaluator_rel_mp) in zip(evaluator_list, evaluator_multiple_preds_list):
        gt_entry_rel = gt_entry.copy()
        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue
        
        evaluator_rel[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )
        evaluator_rel_mp[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )
