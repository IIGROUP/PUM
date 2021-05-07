from lib.word_vectors import obj_edge_vectors
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from config import DATA_PATH
import os
from lib.get_dataset_counts import get_counts, get_counts_ori
from config import cfg









class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, eps=1e-3):
        super(FrequencyBias, self).__init__()

        fg_matrix, bg_matrix = get_counts(must_overlap=True)
        # bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix

        pred_dist = fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps)

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])
        pred_dist_log = torch.nn.functional.log_softmax(Variable(pred_dist), dim=-1).data

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist_log
        self.obj_baseline_state = torch.nn.functional.softmax(Variable(pred_dist.clone()).cuda(), dim=-1)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_labels_state(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline_state[labels[:, 0] * self.num_objs + labels[:, 1]]

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline









class FrequencyBias_ori(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, eps=1e-3):
        super(FrequencyBias_ori, self).__init__()

        fg_matrix, bg_matrix = get_counts_ori(must_overlap=True)
        # fg_matrix, bg_matrix = get_counts_new(must_overlap=True)
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix

        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline