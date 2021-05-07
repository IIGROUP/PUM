import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import operator
import os
import pickle
import numpy as np
from numpy import linalg as la
import gensim
import json
from functools import reduce

from config import cfg, SAMPLING_K, DIM_EMBED, NORM_SCALE, OLD_DATA_PATH


class BiModalRel(nn.Module):
    def __init__(self, dim_in, num_prd_classes):
        super().__init__()

        _, all_prd_vecs = get_obj_prd_vecs('vg')
        self.num_prd_classes = num_prd_classes

        self.prd_vecs = torch.from_numpy(all_prd_vecs.astype('float32')).contiguous().cuda(async=True)
        assert len(self.prd_vecs) == self.num_prd_classes

        self.dim_embed = DIM_EMBED
        self.dim_prd_sem_embed = self.dim_embed
        self.dim_prd_vis_embed = self.dim_embed

        if cfg.use_gaussian:
            self.dim_prd_sem_embed *= 2  # to split as mean and variance
            self.dim_prd_vis_embed *= 2 * self.num_prd_classes  # model visual feature for each predicate

        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.dim_prd_vis_embed))
        self.prd_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.dim_prd_sem_embed))

        if cfg.use_gaussian:
            # Add sigmoid layer
            self.prd_vis_embeddings.add_module('sigmoid', nn.Sigmoid())
            self.prd_sem_embeddings.add_module('sigmoid', nn.Sigmoid())

        self.prd_vis_embed = None
        self.prd_word_embed = None

        self._init_weights()

    def _init_weights(self):

        def XavierFill(tensor):
            """Caffe2 XavierFill Implementation"""
            size = reduce(operator.mul, tensor.shape, 1)
            fan_in = size / tensor.shape[0]
            scale = math.sqrt(3 / fan_in)
            return nn.init.uniform_(tensor, -scale, scale)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rel_feat):

        device_id = rel_feat.get_device()

        self.prd_vis_embed = self.prd_vis_embeddings(rel_feat)  # x^p
        ds_prd_vecs = self.prd_vecs.cuda(device_id, async=True)
        self.prd_word_embed = self.prd_sem_embeddings(ds_prd_vecs)  # y^p
        rel_dists = similarity_score(self.prd_word_embed, self.prd_vis_embed)

        return rel_dists

    def regularization_term(self, labels):
        batch_size = self.prd_vis_embed.shape[0]
        vocab_size = self.prd_word_embed.shape[0]
        pos_index = labels.unsqueeze(-1).unsqueeze(-1).expand([batch_size, 1, self.dim_prd_sem_embed])
        # To compute regularization for positive visual Gaussian embedding
        pos_vis_embed = self.prd_vis_embed.view([batch_size, vocab_size, -1]).gather(1, pos_index).squeeze(1)
        if cfg.gaussian_reg == 'vib':
            kl_prd = kl_reg(self.prd_word_embed)
            kl_vis = kl_reg(pos_vis_embed)
            reg_term = torch.mean(kl_prd) + torch.mean(kl_vis)
            reg_term *= cfg.reg_weight
        elif cfg.gaussian_reg == 'entropy':
            entropy_prd = gaussian_entropy(self.prd_word_embed)
            entropy_vis = gaussian_entropy(pos_vis_embed)
            zero = torch.zeros_like(entropy_prd)
            reg_term = torch.max(cfg.uncer_margin - entropy_prd, zero)[0]\
                       + torch.max(cfg.uncer_margin - entropy_vis, zero)[0]
            reg_term *= cfg.reg_weight
            assert reg_term >= 0
            return reg_term, entropy_prd, entropy_vis
        else:
            raise ValueError('Unexpected reg type')
        return reg_term


def get_obj_prd_vecs(dataset_name='vg'):
    cache_dir = os.path.join(OLD_DATA_PATH, 'cache')
    cache_obj_vecs_path = os.path.join(cache_dir, '%s_obj_vecs.pkl' % dataset_name)
    cache_prd_vecs_path = os.path.join(cache_dir, '%s_prd_vecs.pkl' % dataset_name)

    if os.path.exists(cache_obj_vecs_path) and os.path.exists(cache_prd_vecs_path):
        all_obj_vecs = pickle.load(open(cache_obj_vecs_path, 'rb'))
        all_prd_vecs = pickle.load(open(cache_prd_vecs_path, 'rb'))
        print('Label vectors loaded from cache.')
        return all_obj_vecs, all_prd_vecs

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        OLD_DATA_PATH + '/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
    print('Model loaded.')
    # change everything into lowercase
    all_keys = list(word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
    print('Wiki words converted to lowercase.')

    if dataset_name.find('vrd') >= 0:
        with open(OLD_DATA_PATH + '/vrd/objects.json') as f:
            obj_cats = json.load(f)
        with open(OLD_DATA_PATH + '/vrd/predicates.json') as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vrr-vg') >= 0:
        with open(OLD_DATA_PATH + '/vrr-vg/objects.json') as f:
            obj_cats = json.load(f)
        with open(OLD_DATA_PATH + '/vrr-vg/predicates.json') as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg') >= 0:
        with open(OLD_DATA_PATH + '/vg/objects.json') as f:
            obj_cats = json.load(f)
        with open(OLD_DATA_PATH + '/vg/predicates.json') as f:
            prd_cats = json.load(f)
    else:
        raise NotImplementedError

    word_map = {
        't-shirt': 'tshirt'
    }
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    prd_cats.insert(0, 'unknown')
    all_obj_vecs = np.zeros((len(obj_cats), 300), dtype=np.float32)
    go_wrong = False
    for r, obj_cat in enumerate(obj_cats):
        obj_words = obj_cat.split()
        for word in obj_words:
            if word_map.get(word):
                word = word_map[word]
            try:
                raw_vec = word2vec_model[word]
                all_obj_vecs[r] += (raw_vec / la.norm(raw_vec))
            except KeyError as e:
                print(e)
                go_wrong = True
        all_obj_vecs[r] /= len(obj_words)
    print('Object label vectors loaded.')
    all_prd_vecs = np.zeros((len(prd_cats), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prd_cats):
        prd_words = prd_cat.split()
        for word in prd_words:
            if word_map.get(word):
                word = word_map[word]
            try:
                raw_vec = word2vec_model[word]
                all_prd_vecs[r] += (raw_vec / la.norm(raw_vec))
            except KeyError as e:
                print(e)
                go_wrong = True
        all_prd_vecs[r] /= len(prd_words)
    print('Predicate label vectors loaded.')

    if go_wrong:
        exit(-1)

    pickle.dump(all_obj_vecs, open(cache_obj_vecs_path, 'wb'))
    pickle.dump(all_prd_vecs, open(cache_prd_vecs_path, 'wb'))

    return all_obj_vecs, all_prd_vecs


def similarity_score(word_embed, vis_embed, is_pred=True):
    if cfg.use_gaussian and is_pred:
        batch_size = len(vis_embed)
        word_embed = word_embed.expand([batch_size] + list(word_embed.shape))  # (#bs, #vocab, feat_dim)
        vis_embed = vis_embed.view(word_embed.shape)
        sim_matrix = match_prob_for_sto_embed(word_embed, vis_embed)
    else:
        vis_embed = F.normalize(vis_embed, p=2, dim=1)  # (#bs, feat_dim)
        word_embed = F.normalize(word_embed, p=2, dim=1)  # (#bs, feat_dim)
        sim_matrix = torch.mm(vis_embed, word_embed.t())  # (#bs, #vocab)

        assert sim_matrix.shape == (vis_embed.shape[0], word_embed.shape[0])
    scores = NORM_SCALE * sim_matrix
    return scores


def match_prob_for_sto_embed(sto_embed_word, sto_embed_vis):
    """
    Compute match probability for two stochastic embeddings
    :param sto_embed_word: (batch_size, num_words, hidden_dim * 2)
    :param sto_embed_vis: (batch_size, num_words, hidden_dim * 2)
    :return (batch_size, num_words)
    """
    assert not bool(torch.isnan(sto_embed_word).any()) and not bool(torch.isnan(sto_embed_vis).any())
    batch_size = sto_embed_word.shape[0]
    num_words = sto_embed_word.shape[1]
    mu_word, var_word = torch.split(sto_embed_word, DIM_EMBED, dim=-1)
    mu_vis, var_vis = torch.split(sto_embed_vis, DIM_EMBED, dim=-1)
    if cfg.metric == 'monte-carlo':
        k = SAMPLING_K
        z_word = batch_rsample(mu_word, var_word, k)  # (batch_size, num_words, k, hidden_dim)
        z_vis = batch_rsample(mu_vis, var_vis, k)  # (batch_size, num_words, k, hidden_dim)
        num_samples = k
        z_word = z_word.unsqueeze(3).repeat([1, 1, 1, k, 1])  # (batch_size, num_words, k, k, hidden_dim)
        z_vis = z_vis.repeat([1, 1, k, 1]).reshape(list(z_vis.shape[:2]) + [k, k, -1])  # (batch_size, num_words, k, k, hidden_dim)
        if z_vis.shape[1] == 1:
            z_vis = z_vis.repeat([1, num_words, 1, 1, 1])  # (batch_size, num_words, k, k, hidden_dim)

        # Compute probabilities for all pair combinations
        match_prob = - torch.sqrt(torch.sum((z_word - z_vis) ** 2, dim=-1))
        match_prob = match_prob.sum(-1).sum(-1) / (num_samples ** 2)

        if k > 1 and batch_size > 1 and num_words > 0:
            assert bool(torch.all(z_word[0, 0, 0, 0] == z_word[0, 0, 0, 1]))
            assert bool(torch.all(z_vis[0, 0, 0, 0] == z_vis[0, 0, 1, 0]))
            if sto_embed_vis.shape[1] == 1 and num_words > 1:
                assert bool(torch.all(z_vis[0, 0] == z_vis[0, 1]))
    elif cfg.metric == 'w-distance':
        match_prob = torch.sum((mu_word - mu_vis) ** 2 + (torch.sqrt(var_word) - torch.sqrt(var_vis)) ** 2, dim=-1)
    else:
        raise ValueError('Unexpected metric type')
    assert match_prob.shape == (batch_size, num_words)

    return match_prob


def batch_rsample(mu, var, k):
    """
    Sample from standard normal distribution and use reparameterization trick for k times
    """
    in_shape = list(mu.shape)
    out_shape = in_shape[:-1] + [k, in_shape[-1]]
    mu = mu.unsqueeze(-2).expand(out_shape)
    var = var.unsqueeze(-2).expand(out_shape)
    epsilon = torch.empty(out_shape, device=mu.device).normal_(0, 1.0)
    z = mu + torch.sqrt(var) * epsilon
    return z


def kl_reg(sto_embed):
    """
    Compute KL Divergence between p(z|x) and standard gaussian r(z)
    The formula refers to
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    :return (batch,)
    """
    mu, var = torch.split(sto_embed, DIM_EMBED, dim=1)
    kl = 1/2 * torch.sum(var + mu**2 - torch.log(var) - 1, dim=1)
    return kl


def gaussian_entropy(sto_embed, to_split=True):
    """
    Compute mean differential entropy for a batch of multivariate gaussian distributions
    Refer to https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Differential_entropy
    and paper 'Robust Person Re-Identification by Modelling Feature Uncertainty'
    :return: scalar
    """
    if to_split:
        _, var = torch.split(sto_embed, DIM_EMBED, dim=1)
    else:
        var = sto_embed
    eps = 1e-8
    entropy = float(DIM_EMBED / 2 * (np.log(2*np.pi) + 1)) + torch.sum(torch.log(var + eps), 1) / 2
    mean_entropy = torch.mean(entropy)
    return mean_entropy
