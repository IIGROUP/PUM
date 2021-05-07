import json
import h5py
import os
import pickle
from config import cfg, OLD_DATA_PATH
from dataloaders.visual_genome import VG
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
import cv2
from bounding_box import bounding_box as bb
from collections import defaultdict

matplotlib.use('agg')


def load_pred_entries(path):
    with open(path, 'rb') as f:
        cache = pickle.load(f)
        if isinstance(cache, list):
            return cache
        else:
            return cache['pred_entries']


# load predicted results and GT
pred_cache_dir = '/data/ygc/SGG-Gaussian/checkpoints/predcls-cross_att-vis_gaussian/caches'
inference_a = load_pred_entries(os.path.join(pred_cache_dir, 'test_prediction-0.pkl'))
inference_b = load_pred_entries(os.path.join(pred_cache_dir, 'test_prediction-1.pkl'))
test_data = VG('test', num_val_im=cfg.val_size, filter_duplicate_rels=True,
               use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
               num_im=cfg.num_im)
with open(os.path.join(OLD_DATA_PATH, 'vg', 'predicates.json')) as f:
    prd_label_list = ['__no_relation__'] + json.load(f)  # a list of labels
with open(os.path.join(OLD_DATA_PATH, 'vg', 'objects.json')) as f:
    obj_label_list = ['__background__'] + json.load(f)  # a list of labels


def add_bbox(image, box, label, color):
    left, top, right, bottom = box
    bb.add(image, left, top, right, bottom, label, color)


def scale_img(img):
    height, width = img.shape[:2]
    # scale the image
    smaller_dim = round(1024 * min(width, height) / max(width, height))
    if width > height:
        width = 1024
        height = smaller_dim
    else:
        height = 1024
        width = smaller_dim
    return cv2.resize(img, (width, height))


# get image info by index
def get_info_by_idx(idx, k=20, thres=0.5):
    skip = False
    # image path
    img_path = test_data.filenames[idx]
    # boxes
    boxes = test_data.gt_boxes[idx]
    # object labels
    idx2label = obj_label_list
    labels = []
    labels_count = defaultdict(int)
    for i, label_idx in enumerate(test_data.gt_classes[idx]):
        label = idx2label[label_idx]
        labels.append(label + str(labels_count[label]))
        labels_count[label] += 1

    if len(labels) > 10:
        skip = True

    # groundtruth relation triplet
    idx2pred = prd_label_list

    def get_pred_rels_by_idx(det_input):
        # prediction relation triplet
        pred_rel_pair = det_input[idx]['pred_rel_inds']
        pred_rel_label = 1 + det_input[idx]['rel_scores'][:, 1:].argmax(1)
        return [(labels[i[0]], idx2pred[j], labels[i[1]])
                for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]

    pred_rels_a = get_pred_rels_by_idx(inference_a)[:k]
    pred_rels_b = get_pred_rels_by_idx(inference_b)[:k]

    gt_rels = test_data.relationships[idx]
    gt_rels = [(labels[i[0]], idx2pred[i[2]], labels[i[1]]) for i in gt_rels]
    gt_rels = list(set(gt_rels))

    # Combine two inferences
    pair2prd_a = {}
    for pred_rel_a in pred_rels_a:
        pair2prd_a[(pred_rel_a[0], pred_rel_a[2])] = pred_rel_a[1]
    pair2prd_gt = {}
    for rel in gt_rels:
        pair2prd_gt[(rel[0], rel[2])] = rel[1]

    pred_rels = []
    count_diff = 0  # count how many predicates are different
    for pred_rel_b in pred_rels_b:
        sbj = pred_rel_b[0]
        obj = pred_rel_b[2]
        so = (sbj, obj)
        prd_b = pred_rel_b[1]
        if so not in pair2prd_a or so not in pair2prd_gt:
            continue
        prd_a = pair2prd_a[so]
        prd_gt = pair2prd_gt[so]
        if prd_a != prd_b and prd_gt in [prd_a, prd_b]:
            pred_rels.append((sbj, '%s/%s' % (prd_a, prd_b), obj))
            count_diff += 1
        else:
            pred_rels.append(pred_rel_b)

    if count_diff < 2:
        skip = True

    if len(gt_rels) < 5:
        skip = True

    if skip:
        img_path = None  # indicate not to show this image

    return img_path, boxes, labels, gt_rels, pred_rels


def draw_single_box(pic, box, color='green', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)


def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))


def draw_image(img_path, boxes, labels, gt_rels, pred_rels, print_img=True):
    if img_path is None:
        return 0
    img_name = os.path.basename(img_path)
    if not img_name == '2318942.jpg':
        return 0
    print('\n\n' + img_name)
    pic = cv2.imread(img_path)
    pic = scale_img(pic)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = labels[i]
        add_bbox(pic, boxes[i], info, 'green')
    #         draw_single_box(pic, boxes[i], draw_info=info)
    if print_img:
        save_dir = '/data/ygc/SGG-Gaussian/det_results'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, img_name), pic)
        plt.axis('off')
        pic = pic[:, :, ::-1]  # Convert BGR (OpenCV style) to RGB
        fig = plt.gcf()
        plt.imshow(pic)
        plt.show()
    if print_img:
        pred_diff = [x for x in pred_rels if '/' in x[1]]
        print_list('pred_diff', pred_diff)
        print('*' * 50)
        print_list('pred_rels', pred_rels)
        print('*' * 50)
        print_list('gt_boxes', labels)
        print('*' * 50)
        print_list('gt_rels', gt_rels)
        print('*' * 50)
    return 1


def show_all(length):
    count = 0
    for cand_idx in range(len(test_data)):
        count += draw_image(*get_info_by_idx(cand_idx))
        if count == length:
            break


show_all(1000)
