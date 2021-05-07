"""
Training script for scene graph detection. Integrated with Rowan's faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG, vg_collate
import numpy as np
from torch import optim, nn
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import pandas as pd
import time
import os
from tqdm import tqdm
from glob import glob
import traceback

from config import cfg, BOX_SCALE, IM_SCALE, PREDICATES_WEIGHTS
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para, nps_loss, focal_loss
from lib.utils import load_ckpt, define_model, do_test
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

pred_weight = None
if cfg.pred_weight:
    pred_weight = torch.from_numpy(PREDICATES_WEIGHTS).float().cuda()

# We use tensorboard to observe results and decrease learning rate manually. If you want to use TB, you need to install TensorFlow fist.
if cfg.tb_log_dir is not None:
    from tensorboardX import SummaryWriter

    if not os.path.exists(cfg.tb_log_dir):
        os.makedirs(cfg.tb_log_dir)
    writer = SummaryWriter(log_dir=cfg.tb_log_dir)
    use_tb = True
else:
    use_tb = False

train, val, test = VG.splits(num_val_im=cfg.val_size, filter_duplicate_rels=True,
                          use_proposals=cfg.use_proposals,
                          filter_non_overlap=cfg.mode == 'sgdet',
                          num_im=cfg.num_im)

ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               num_gpus=cfg.num_gpus)

detector = define_model(cfg, train.ind_to_classes, train.ind_to_predicates)

# Freeze the detector
if hasattr(detector, 'detector'):
    for n, param in detector.detector.named_parameters():
        param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n, p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params, 'lr': lr}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if cfg.adam:
        optimizer = optim.Adam(params, weight_decay=cfg.adamwd, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=cfg.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    if cfg.new_lr_strategy:
        milestones = [7]
    else:
        milestones = [7, 12, 15]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler


def save_model(save_name=None):
    if save_name is None:
        save_name = '{}-{}.tar'.format('vgrel', epoch)
    if cfg.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(cfg.save_dir, save_name))


def train_epoch(epoch_num):
    detector.train()
    losses = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        if cfg.use_rl_tree:
            batch_train = train_batch_rl
        else:
            batch_train = train_batch
        res = batch_train(batch, b, verbose=b % (cfg.print_interval * 10) == 0)
        losses.append(res['losses'])

        if b % cfg.print_interval == 0 and b >= cfg.print_interval:
            mn = pd.concat(losses[-cfg.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / cfg.print_interval
            print("\n[{}] epoch: {:2d}, batch: {:5d}/{:5d}, {:.3f}s/batch, {:.1f}m/epoch".format(
                cfg.run_name, epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            if use_tb:
                writer.add_scalar('train/total_loss', mn[-1], epoch_num)
                writer.add_scalar('lr/lr0', optimizer.param_groups[0]['lr'], epoch_num)
                writer.add_scalar('lr/lr1', optimizer.param_groups[1]['lr'], epoch_num)
                writer.file_writer.flush()
            print('-----------', flush=True)
            start = time.time()
    return {
        'losses': pd.concat(losses, axis=1),
    }


def input_two_gt(gt_classes, gt_relationships, gt_boxes):
    gt_classes = gt_classes[:,1].contiguous().view(-1).data.cpu().numpy()
    gt_relationships = gt_relationships[:,1:].data.cpu().numpy()
    gt_boxes = (gt_boxes * BOX_SCALE/IM_SCALE).data.cpu().numpy()
    return gt_classes, gt_relationships, gt_boxes


def get_recall_x(batch_num, batch, det_res, evaluator, x=100):
    gt_boxes, gt_classes, gt_rels = batch.gt_boxes, batch.gt_classes, batch.gt_rels
    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_classes_i, gt_rels_i, gt_boxes_i = input_two_gt(gt_classes, gt_rels, gt_boxes)
        gt_entry = {
            'gt_classes': gt_classes_i.copy(),
            'gt_relations': gt_rels_i.copy(),
            'gt_boxes': gt_boxes_i.copy(),
        }

        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[cfg.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    return evaluator[cfg.mode].result_dict[cfg.mode + '_recall'][x]


def fix_batchnorm(model):
    if isinstance(model, list):
        for m in model:
            fix_batchnorm(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                #print('Fix BatchNorm1d')
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                #print('Fix BatchNorm2d')
                m.eval()
            elif isinstance(m, nn.BatchNorm3d):
                #print('Fix BatchNorm3d')
                m.eval()
            elif isinstance(m, nn.Dropout):
                #print('Fix Dropout')
                m.eval()
            elif isinstance(m, nn.AlphaDropout):
                #print('Fix AlphaDropout')
                m.eval()


def fix_rest_net(model):
    for n, param in model.named_parameters():
        param.requires_grad = False
    for n, param in model.context.feat_preprocess_net.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_sub.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_obj.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_scores.named_parameters():
        param.requires_grad = True
    # fix batchnorm during self critic
    fix_batchnorm(model)


def fix_tree_score_net(model):
    for n, param in model.context.feat_preprocess_net.named_parameters():
        param.requires_grad = False
    for n, param in model.context.rl_sub.named_parameters():
        param.requires_grad = False
    for n, param in model.context.rl_obj.named_parameters():
        param.requires_grad = False
    for n, param in model.context.rl_scores.named_parameters():
        param.requires_grad = False
    for n, param in model.context.obj_embed.named_parameters():
        param.requires_grad = False
    for n, param in model.context.virtual_node_embed.named_parameters():
        param.requires_grad = False
    fix_batchnorm(model)


def cal_policy_gradient_loss(loss_container, current_reward, base_reward):
    if len(loss_container) == 0:
        return Variable(torch.FloatTensor([0]).cuda()).view(-1)
    else:
        return (base_reward - current_reward) * sum(loss_container) / len(loss_container)


def train_batch_rl(b, count, verbose=False,
                   base_evaluator=BasicSceneGraphEvaluator.all_modes(),
                   train_evaluator=BasicSceneGraphEvaluator.all_modes()):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    detector.eval()
    base_eval = detector[b]
    base_reward = float(get_recall_x(count, b, [base_eval], base_evaluator, 100)[-1])
    del base_eval

    detector.rl_train = True
    detector.train()
    fix_batchnorm(detector)
    SAMPLE_NUM = 5

    for k in range(SAMPLE_NUM):
        result, train_eval = detector[b]
        current_reward = float(get_recall_x(count, b, [train_eval], train_evaluator, 100)[-1])
        del train_eval

        losses = {}

        if base_reward == current_reward or float(sum(result.gen_tree_loss)) == 0:
            losses['policy_gradient_gen_tree_loss'] = 0
            loss = 0
            continue

        # policy gradient loss
        losses['policy_gradient_gen_tree_loss'] = cal_policy_gradient_loss(result.gen_tree_loss, current_reward,
                                                                           base_reward)

        loss = sum(losses.values()) / SAMPLE_NUM
        loss.backward()
        del result
    detector.rl_train = False
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=cfg.clip, verbose=verbose, clip=True)
    optimizer.step()
    optimizer.zero_grad()
    losses['total'] = loss
    res = {
        'losses': pd.Series({x: (y.data if isinstance(y, torch.Tensor) else y) for x, y in losses.items()}),
    }
    return res


def train_batch(b, index, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}

    if cfg.use_obj:  # if not use ggnn obj, we just use scores of faster rcnn as their scores, there is no need to train
        if cfg.use_nps_loss:
            losses['class_loss'] = nps_loss(result.rm_obj_dists, result.rm_obj_labels, result.rel_labels, result.im_inds)
        if cfg.use_focal_loss:
            losses['class_loss'] = focal_loss(result.rm_obj_dists, result.rm_obj_labels)
        else:
            losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        losses['class_loss'] *= cfg.class_loss_weight

    if cfg.use_bimodal_rel and cfg.gaussian_reg != '':
        reg_ret = detector.rel_compress.regularization_term(result.rel_labels[:, -1])
        if cfg.gaussian_reg == 'entropy':
            losses['gaussian_reg'], entropy_prd, entropy_vis = reg_ret
        else:
            losses['gaussian_reg'] = reg_ret

    if cfg.mode not in ['objcls', 'objdet'] and not cfg.no_rel_loss:
        losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1], weight=pred_weight)
        if cfg.visual_gaussian:
            batch_size = result.rel_labels[:, -1].shape[0]
            rel_labels = result.rel_labels[:, -1].unsqueeze(-1).\
                expand([batch_size, cfg.num_gaussian_samples]).flatten()  # (B*N, )
            losses['rel_z_loss'] = F.cross_entropy(result.rel_dists_z, rel_labels, weight=pred_weight) * cfg.sampling_loss_weight
            losses['rel_loss'] = losses['rel_loss'] * (1.0 - cfg.sampling_loss_weight)
            # add regularization
            losses['gaussian_reg'], _ = detector.regularization_term()

    if cfg.model == 'lsvru':
        # The indexes of sbj/obj correspond to that in `obj_preds`
        # The indexes of `obj_preds` start from 1, so we subtract 1
        sbj_labels = result.obj_preds[result.rel_labels[:, 1]] - 1
        obj_labels = result.obj_preds[result.rel_labels[:, 2]] - 1
        losses['sbj_loss'] = F.cross_entropy(result.sbj_dists, sbj_labels)
        losses['obj_loss'] = F.cross_entropy(result.obj_dists, obj_labels)
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=cfg.clip, verbose=verbose, clip=True)
    assert not torch.any(torch.isnan(loss))
    losses['total'] = loss
    optimizer.step()
    res = {
        'losses': pd.Series({x: y.data for x, y in losses.items()}),
    }
    return res


def val_epoch():
    detector.eval()
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes()  # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(cfg.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                  evaluator_multiple_preds_list)

    return evaluator[cfg.mode].print_stats()


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    det_res = detector[b]
    if cfg.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(cfg.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)


if cfg.model == 'vctree':
    if cfg.use_rl_tree:
        fix_rest_net(detector)
    else:
        fix_tree_score_net(detector)
start_epoch, optimizer, scheduler = load_ckpt(detector, cfg.ckpt)
if optimizer is None:
    optimizer, scheduler = get_optim(cfg.lr * cfg.num_gpus * cfg.batch_size)
detector.cuda()

print("Training starts now!")

best_recall, best_mean_recall = 0, 0
for epoch in range(start_epoch + 1, start_epoch + 1 + cfg.num_epochs):
    try:
        lr_set = set([pg['lr'] for pg in optimizer.param_groups])
        print('lr set: ', lr_set)
        rez = train_epoch(epoch)
        losses = rez['losses'].mean(1)
        print("overall{:2d}: ({:.3f})\n{}".format(epoch, losses['total'], losses), flush=True)

        if use_tb:
            for k, v in losses.items():
                writer.add_scalar('loss/%s' % k, v, epoch)

        if (epoch+1) % cfg.save_freq == 0 or epoch == start_epoch + cfg.num_epochs:
            save_model()

        torch.cuda.empty_cache()
        with torch.no_grad():
            recall, mean_recall, _, acc = val_epoch()
        if use_tb:
            for key, value in recall.items():
                writer.add_scalar('eval_' + cfg.mode + '_with_constraint/' + key, value, epoch)
            for key, value in mean_recall.items():
                writer.add_scalar('eval_' + cfg.mode + '_with_constraint/mean ' + key, value, epoch)
            if acc:
                writer.add_scalar('obj_cls_acc', acc, epoch)

        if cfg.test_as_val:
            # Save the best model
            recall = recall['R@100']
            mean_recall = mean_recall['mR@100']
            if recall > best_recall:
                best_recall = recall
                print('Save best-recall (%.2f) model' % (recall * 100))
                save_model('best_recall.tar')
            if mean_recall > best_mean_recall:
                best_mean_recall = mean_recall
                print('Save best-mean-recall (%.2f) model' % (mean_recall * 100))
                save_model('best_mean_recall.tar')

        scheduler.step()
        if any([pg['lr'] <= (cfg.lr * cfg.num_gpus * cfg.batch_size) / 9999.0 for pg in optimizer.param_groups]):
            print("exiting training early")
            save_model()
            writer.close()
            break
    except Exception as e:
        traceback.print_exc()
        # Save model in case that we have to interrupt suddenly if trained long enough
        if epoch > 1:
            print('Saving model due to exception')
            save_model()
        del train_loader
        exit()

# Remove old model checkpoints
if not cfg.keep_old_ckpt:
    for ckpt_path in glob(os.path.join(cfg.save_dir, 'vgrel-*.tar')):
        ckpt_epoch = int(ckpt_path.split('-')[-1].split('.')[0])
        if ckpt_epoch < epoch:
            try:
                os.remove(ckpt_path)
            except Exception as e:
                print('fail to remove %s due to %s' % (ckpt_path, e))

# Test model
print('Start to do testing')
torch.cuda.empty_cache()
test_loader = DataLoader(
    dataset=test,
    batch_size=cfg.num_gpus,
    shuffle=False,
    num_workers=cfg.num_workers,
    collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=cfg.num_gpus, is_train=False),
    drop_last=True,
    pin_memory=True,
)
do_test(detector, test, test_loader)
