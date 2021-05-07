#!/usr/bin/python
# -*- coding: utf-8 -*-
# visualization code for categorical recall difference between two models
"""
Example:
    python visualization/visualize_recall_diff.py \
    --baseline_result ../VCTree-Scene-Graph-Generation/caches/prd_recall.pkl \
    --new_model_result checkpoints/predcls-cross_att-vis_gaussian/caches/test_prediction-0.pkl
"""
import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import csv


matplotlib.use('Agg')
OLD_DATA_PATH = '../Large-Scale-VRD.pytorch/data/'

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_result', type=str,
        default='../VCTree-Scene-Graph-Generation/caches/prd_recall.pkl',
        help='path of baseline result')
    parser.add_argument(
        '--new_model_result', type=str,
        default='checkpoints/predcls-cross_att-vis_gaussian/caches/test_prediction-0.pkl',
        help='path of new model result')
    parser.add_argument(
        '--k', type=int,
        default=100,
        help='recall@k to visualize')

    return parser.parse_args()


def parse_category_test_results(file_path):
    data = pickle.load(open(file_path, 'rb'))
    if 'eval_results' in data:
        eval_res = data['eval_results']['prd_recall']
    else:
        eval_res = data
    prd2recall = eval_res[args.k]
    return prd2recall


def sort_dict(dict_in, key=lambda item: item[1]):
    return {k: v for k, v in sorted(dict_in.items(), key=key, reverse=True)}


def autolabel(ax, rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    refer to https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """
    fixed_args = dict(
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom',
        rotation=90,
        fontsize=14
    )
    for rect in rects:
        height = rect.get_height()
        if height >= 0:
            text = '+%.2f' % height
            y = height
        else:
            text = '%.2f' % height
            y = 0
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, y),
                    **fixed_args)


def rgb_values_0_to_1(r, g, b):
    return r / 255, g / 255, b / 255


def draw_bar_figure(data, save_path):
    # TODO there may be something wrong with the predicate counts
    with open(os.path.join(OLD_DATA_PATH, 'cache/vg_prd_count.pkl'), 'rb') as f:
        prd_counts = pickle.load(f)
    prd_counts = sort_dict(prd_counts)
    with open(os.path.join(OLD_DATA_PATH, 'vg/predicates.json')) as f:
        prd_label_list = json.load(f)  # a list of labels
    names = []
    nums = []
    for k in prd_counts.keys():
        name = prd_label_list[k]
        num = data[k+1] * 100
        # Filter out trivial values
        if abs(num) > 0.01:
            names.append(name)
            nums.append(num)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.yaxis.grid(zorder=0)  # Set z-order to make sure the gridlines behind the bars
    rects = bar_list = ax.bar(names, nums, zorder=3)
    print('y data', nums)
    for i in range(len(bar_list)):
        if nums[i] < 0:
            bar_list[i].set_color(rgb_values_0_to_1(237, 125, 49))
        else:
            bar_list[i].set_color(rgb_values_0_to_1(68, 114, 196))
    plt.xticks(np.arange(len(names)), names, rotation=90)
    # Set parameters for tick labels
    plt.tick_params(axis='x', which='major', labelsize=21)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.ylabel('R@%d Improvement (%%)' % args.k, fontsize=23)
    plt.ylim((None, 16))  # a trick to to make sure the texts are all inside the figure
    # Show number above the bars
    autolabel(ax, rects)
    plt.tight_layout()
    plt.savefig(save_path + '.pdf')
    print('figure saved at', save_path + '.pdf')
    plt.close()


def save_as_csv(baseline_data, new_data, save_path):
    with open(os.path.join(OLD_DATA_PATH, 'cache/vg_prd_count.pkl'), 'rb') as f:
        prd_counts = pickle.load(f)
    prd_counts = sort_dict(prd_counts)
    with open(os.path.join(OLD_DATA_PATH, 'vg/predicates.json')) as f:
        prd_label_list = json.load(f)  # a list of labels
    writer = csv.writer(open(save_path + '.csv', 'w'))
    writer.writerow(['', 'baseline', 'new'])  # writer headers
    for k in prd_counts.keys():
        name = prd_label_list[k]
        writer.writerow([name, baseline_data[k+1] * 100, new_data[k+1] * 100])
    print('csv saved at', save_path + '.csv')


if __name__ == '__main__':
    args = parse_args()
    assert args.new_model_result != ''
    if args.baseline_result == '':
        # Use a default dict where every value is 0
        baseline_prd2recall = defaultdict(int)
    else:
        baseline_prd2recall = parse_category_test_results(args.baseline_result)
    new_model_prd2recall = parse_category_test_results(args.new_model_result)
    figure_name = '%s-vs-%s' % \
                  (args.new_model_result.split('/')[-3], args.baseline_result.split('/')[-3] if args.baseline_result else '')
    save_base_name = os.path.join(os.path.dirname(args.new_model_result), figure_name)
    save_as_csv(baseline_prd2recall, new_model_prd2recall, save_base_name)
    recall_diff = {k: v - baseline_prd2recall[k] for k, v in new_model_prd2recall.items()}
    draw_bar_figure(recall_diff, save_base_name)
