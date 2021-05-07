import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import cv2
from bounding_box import bounding_box as bb
from dataloaders.visual_genome import VG
from config import cfg, OLD_DATA_PATH
import json
import pickle
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    "pgf.preamble": [
        r'\usepackage{color}'     # xcolor for colours
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

matplotlib.use('pgf')  # Important


def load_pred_entries(path):
    with open(path, 'rb') as f:
        cache = pickle.load(f)
        if isinstance(cache, list):
            return cache
        else:
            return cache['pred_entries']


def add_bbox(image, box, label, color):
    left, top, right, bottom = box
    bb.add(image, left, top, right, bottom, label, color)


def get_sbj_obj2prd(rels):
    sbj_obj2prd = {}
    for rel in rels:
        sbj_obj2prd[(rel[0], rel[1])] = rel[2]
    return sbj_obj2prd


def visualize_multiple_inferences(img_ind, sbj_obj, label_dict, save_path):
    img_path = test_data.filenames[img_ind]
    boxes = test_data.gt_boxes[img_ind]
    det_labels = [obj_label_list[x] for x in test_data.gt_classes[img_ind]]
    image = cv2.imread(img_path)
    #     print('original size:', image.shape)
    height, width = image.shape[:2]

    # scale the image
    smaller_dim = round(1024 * min(width, height) / max(width, height))
    if width > height:
        width = 1024
        height = smaller_dim
    else:
        height = 1024
        width = smaller_dim
    image = cv2.resize(image, (width, height))
    sbj_ind = sbj_obj[0]
    obj_ind = sbj_obj[1]

    add_bbox(image, boxes[sbj_ind], None, 'green')
    add_bbox(image, boxes[obj_ind], None, 'red')

    image = cv2.resize(image, (800, 600))  # Resize to fixed resolution

    image = image[:, :, ::-1]  # Convert BGR (OpenCV style) to RGB

    fig, ax = plt.subplots(figsize=[10, 8])
    # Create two subplots and unpack the output array immediately
    ax.imshow(image)
    # show text information
    fig_width, fig_height = fig.get_size_inches() * fig.dpi  # size in pixels
    text_x = fig_width * 1.1
    text_y = fig_height * 0.5
    text = \
        '%s{%s{%s}} - ? - %s{%s{%s}}\n%s\n' %\
        (r'\textcolor{green}', r'\texttt', det_labels[sbj_ind], r'\textcolor{red}', r'\texttt', det_labels[obj_ind], r'\rule{5cm}{0.4pt} ') + \
        '\n'.join(['%s: %s{%s}' % (k, r'\quad \texttt', v) for k, v in label_dict.items()])
    ax.text(text_x, text_y, text, fontsize=15)
    ax.axis('off')
    plt.tight_layout()
    if hit:
        save_path = os.path.join(os.path.dirname(save_path), 'hit-' + os.path.basename(save_path))
    plt.savefig(save_path)
    print('figure saved at', save_path)
    # Save image data and text for later restoring
    to_save = {
        'image': image,
        'text': text,
        'text_x': text_x,
        'text_y': text_y
    }
    pickle.dump(to_save, open(save_path.replace('pdf', 'pkl'), 'wb'))
    plt.close()


def underline(pred, gt):
    if pred == gt:
        return '%s{%s}' % (r'\underline', pred)
    return pred


if __name__ == '__main__':

    with open(os.path.join(OLD_DATA_PATH, 'vg', 'predicates.json')) as f:
        prd_label_list = ['__no_relation__'] + json.load(f)  # a list of labels
    with open(os.path.join(OLD_DATA_PATH, 'vg', 'objects.json')) as f:
        obj_label_list = ['__background__'] + json.load(f)  # a list of labels

    test_data = VG('test', num_val_im=cfg.val_size, filter_duplicate_rels=True,
                   use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
                   num_im=cfg.num_im)

    pred_cache_dir = '/data/ygc/SGG-Gaussian/checkpoints/predcls-cross_att-vis_gaussian/caches'
    inference_a = load_pred_entries(os.path.join(pred_cache_dir, 'test_prediction-0.pkl'))
    inference_b = load_pred_entries(os.path.join(pred_cache_dir, 'test_prediction-1.pkl'))
    inference_c = load_pred_entries(os.path.join(pred_cache_dir, 'test_prediction-2.pkl'))
    inference_bs = load_pred_entries('/data/ygc/SGG-Gaussian/checkpoints/predcls-cross_att/caches/test_prediction.pkl')

    consider_no_rel = False
    num_diff_rels = 0
    num_total_rels = 0
    num_correct_a = 0
    num_correct_b = 0
    num_correct_c = 0
    res_dir = 'visualization/multiple_inferences_viz'
    os.makedirs(res_dir, exist_ok=True)
    count = 0
    for i, (pred_a, pred_b, pred_c) in enumerate(zip(inference_a, inference_b, inference_c)):
        prd_a = pred_a['rel_scores'].argmax(1) if consider_no_rel else 1 + pred_a['rel_scores'][:, 1:].argmax(1)
        prd_b = pred_b['rel_scores'].argmax(1) if consider_no_rel else 1 + pred_b['rel_scores'][:, 1:].argmax(1)
        prd_c = pred_c['rel_scores'].argmax(1) if consider_no_rel else 1 + pred_c['rel_scores'][:, 1:].argmax(1)
        prd_bs = inference_bs[i]['rel_scores'].argmax(1) if consider_no_rel else 1 + inference_bs[i]['rel_scores'][:, 1:].argmax(1)
        sbj_obj_a = pred_a['pred_rel_inds']  # sbj/obj pairs not consistent between two predictions
        det_classes = pred_a['pred_classes']
        gt_rels = test_data.relationships[i]

        sbj_obj2prd_gt = get_sbj_obj2prd(gt_rels)
        sbj_obj2prd_a = get_sbj_obj2prd(
            np.column_stack((pred_a['pred_rel_inds'], prd_a))
        )
        sbj_obj2prd_b = get_sbj_obj2prd(
            np.column_stack((pred_b['pred_rel_inds'], prd_b))
        )
        sbj_obj2prd_c = get_sbj_obj2prd(
            np.column_stack((pred_c['pred_rel_inds'], prd_c))
        )
        sbj_obj2prd_bs = get_sbj_obj2prd(
            np.column_stack((inference_bs[i]['pred_rel_inds'], prd_bs))
        )

        # Based on only GT pairs
        num_total_rels += len(gt_rels)
        for sbj_obj, cur_prd_gt in sbj_obj2prd_gt.items():
            cur_prd_a = sbj_obj2prd_a[sbj_obj]
            cur_prd_b = sbj_obj2prd_b[sbj_obj]
            cur_prd_c = sbj_obj2prd_c[sbj_obj]
            # Only show situations where two inference results are different
            # But they may be both wrong
            if cur_prd_a != cur_prd_b and cur_prd_a != cur_prd_c and cur_prd_b != cur_prd_c:
                sbj = sbj_obj[0]
                obj = sbj_obj[1]
                sbj_label = obj_label_list[det_classes[sbj]]
                obj_label = obj_label_list[det_classes[obj]]
                cur_prd_bs = sbj_obj2prd_bs[sbj_obj]
                if cur_prd_a == cur_prd_gt or cur_prd_b == cur_prd_gt or cur_prd_c == cur_prd_gt:
                    hit = True
                else:
                    hit = False

                label_a = prd_label_list[cur_prd_a]
                label_b = prd_label_list[cur_prd_b]
                label_c = prd_label_list[cur_prd_c]
                label_bs = prd_label_list[cur_prd_bs]
                label_gt = prd_label_list[cur_prd_gt]
                img_path = test_data.filenames[i]
                print('[%d]: %s' % (count, img_path))
                print('%-8s- %-8s: %-10s v.s. %-10s v.s. %-10s, baseline: %-10s, GT: %-10s%s' %
                      (sbj_label, obj_label, label_a, label_b, label_c, label_bs, label_gt, '*' if hit else ''))
                save_path = os.path.join(res_dir, '%d.pdf' % count)
                count += 1
                label_dict = {
                    'Prediction 1': underline(label_a, label_gt),
                    'Prediction 2': underline(label_b, label_gt),
                    'Prediction 3': underline(label_c, label_gt),
                }
                visualize_multiple_inferences(i, sbj_obj, label_dict, save_path)
                num_diff_rels += 1
                if cur_prd_a == cur_prd_gt:
                    num_correct_a += 1
                if cur_prd_b == cur_prd_gt:
                    num_correct_b += 1
                if cur_prd_c == cur_prd_gt:
                    num_correct_c += 1

    print('%d / %d (%f %%) are diverse' % (num_diff_rels, num_total_rels, (num_diff_rels / num_total_rels * 100)))
    print('A correct: %d / %d (%f %%)' % (num_correct_a, num_total_rels, (num_correct_a / num_total_rels * 100)))
    print('B correct: %d / %d (%f %%)' % (num_correct_b, num_total_rels, (num_correct_b / num_total_rels * 100)))
    print('C correct: %d / %d (%f %%)' % (num_correct_c, num_total_rels, (num_correct_c / num_total_rels * 100)))
    num_correct_either = num_correct_a + num_correct_b + num_correct_c  # We only take the controversial situations
    print('Either correct: %d / %d (%f %%)' % (num_correct_either, num_total_rels, (num_correct_either / num_total_rels * 100)))
