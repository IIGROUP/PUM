"""
Configuration file!
"""
import os
import sys
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
OLD_DATA_PATH = '../Large-Scale-VRD.pytorch/data/'
CO_OCCOUR_PATH = os.path.join(DATA_PATH, 'co_occour_count.npy')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = 'data/datasets/visual-genome/VG_100K'
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls', 'objcls', 'objdet')

LOG_SOFTMAX = True

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

NORM_SCALE = 10.0
SAMPLING_K = 8
DIM_EMBED = 300
VAL_BATCH_SPLIT_SIZE = 256

PREDICATES_WEIGHTS = np.ones(51, dtype=np.float32)
PREDICATES_WEIGHTS[0] = 0.1

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.adamwd = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.mode = None
        self.test = False
        self.adam = False
        self.cache = None
        self.use_proposals=False
        self.use_resnet=False
        self.num_epochs=None
        self.pooling_dim = None

        self.use_obj = False
        self.obj_time_step_num = None
        self.obj_hidden_dim = None
        self.obj_output_dim = None
        self.use_obj_knowledge = False
        self.obj_knowledge = None

        self.use_dualResGCN_rel = False
        self.dualResGCN_rel_hidden_dim = None
        self.dualResGCN_rel_output_dim = None
        self.use_rel_knowledge = False
        self.rel_knowledge = None

        self.tb_log_dir = None
        self.save_rel_recall = None

        self.parser = self.setup_parser()
        args, unknown = self.parser.parse_known_args()
        self.args = vars(args)

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if self.run_name != '':
            self.save_dir = os.path.join(ROOT_PATH, 'checkpoints', self.run_name)
            self.tb_log_dir = os.path.join(ROOT_PATH, 'summaries', self.run_name)
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.tb_log_dir, exist_ok=True)
        else:
            if self.ckpt is not None and self.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
                self.save_dir = os.path.dirname(self.ckpt)
            elif len(self.save_dir) == 0:
                self.save_dir = None
            else:
                self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)

            if len(self.tb_log_dir) != 0:
                self.tb_log_dir = os.path.join(ROOT_PATH, self.tb_log_dir)
                if not os.path.exists(self.tb_log_dir):
                    os.makedirs(self.tb_log_dir) # help make multi depth directories, such as summaries/kern_predcls
            else:
                self.tb_log_dir = None

        if self.cache == '' and self.save_dir is not None:
            self.cache = os.path.join(self.save_dir, 'caches/test_prediction.pkl')
            os.makedirs(os.path.dirname(self.cache), exist_ok=True)
        elif self.cache == 'none':
            self.cache = None

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

        log_mesg = ''
        # Record the current script command
        log_mesg += '~~~~~~~~ Script: ~~~~~~~\n'
        log_mesg += 'python %s\n' % ' '.join(sys.argv)
        log_mesg += '~~~~~~~~ Hyperparameters used: ~~~~~~~\n'
        for x, y in self.__dict__.items():
            log_mesg += '{} : {}\n'.format(x, y)
        log_mesg += '~~~~~~~~ Unknown args: ~~~~~~~\n'
        log_mesg += '{}\n'.format(unknown)
        print(log_mesg)
        if self.save_dir is not None:
            with open(os.path.join(self.save_dir, 'config-%s.txt' % os.path.basename(sys.argv[0])), 'w') as f:
                f.write(log_mesg)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')

        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='checkpoints/vgdet/vg-faster-rcnn.tar')
        parser.add_argument('-run_name', dest='run_name', help='Name of the current run', type=str, default='')
        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)
        parser.add_argument('-resume_training', help='resume for continuing training', action='store_true')
        parser.add_argument('-keep_old_ckpt', help='not to remove all old checkpoints, mainly for finetune debug', action='store_true')

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=1)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=8)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-5)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=8)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay of SGD', type=float, default=1e-4)
        parser.add_argument('-adamwd', dest='adamwd', help='weight decay of adam', type=float, default=0.0)

        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode in {sgdet, sgcls, predcls}', type=str, default='predcls')

        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-cache2', dest='cache2', help='predictions cache path of baseline model for comparison', type=str,
                            default='')

        parser.add_argument('-model', dest='model', help='which model to use', type=str,
                            default='motifs')

        parser.add_argument('-adam', dest='adam', help='use adam', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')

        parser.add_argument('-nimg', dest='num_im', help='Number of images to use, mainly for quick debugging', type=int, default=-1)
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=20)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-save_freq', dest='save_freq', help='Every n epochs to save model', type=int, default=5)
        parser.add_argument('-class_loss_weight', help='discount weight for class loss', type=float, default=1.0)
        parser.add_argument('-use_pred_entries_cache', help='use cache of pred entries instead of running network in evaluation', action='store_true')
        parser.add_argument('-gt_labels_for_bias', help='use GT labels to retrieve bias in training', action='store_true')
        parser.add_argument('-use_word_vec', help='use word vectors in object feature', action='store_true')
        parser.add_argument('-cache_obj_dists', help='cache object predictions', action='store_true')
        parser.add_argument('-cache_det_res', help='cache object detection results', action='store_true')
        parser.add_argument('-cache_gaussians', help='cache predicted gaussians of instances', action='store_true')
        parser.add_argument('-obj_dists_path', help='path of cached object predictions for predcls to read', type=str, default='')
        parser.add_argument('-obj_det_path', help='path of cached object detections for predcls to read', type=str, default='')
        parser.add_argument('-gaussians_path', help='path of cached gaussians', type=str, default='')
        parser.add_argument('-test_data_name', help='split name of test data, could be train sometimes', type=str, default='test')
        parser.add_argument('-inference_times', help='times of inference for visual gaussian model', type=int, default=5)
        parser.add_argument('-no_word_vec_for_predcls', help='not to use word vec in predcls mode, mainly for testing sgcls/sgdet models', action='store_true')
        parser.add_argument('-num_boxes_per_img', help='number of predicted boxes for each image, used for detection', type=int, default=64)
        parser.add_argument('-test_as_val', help='use test data as validation after each epoch', action='store_true')
        parser.add_argument('-new_lr_strategy', help='new lr strategy including only 2 stages', action='store_true')
        parser.add_argument('-use_nps_loss', help='use nps loss for object detection', action='store_true')
        parser.add_argument('-use_focal_loss', help='use focal loss for object detection', action='store_true')
        parser.add_argument('-obj_dists_cache_as_output', help='use cached rm_obj_dists as output without obj_cls', action='store_true')
        parser.add_argument('-add_ori_obj_dists', help='add original rm_obj_dists to the final one', action='store_true')
        parser.add_argument('-fixed_obj_det_in_training', help='use fixed obj det in training', action='store_true')
        parser.add_argument('-no_rel_loss', help='not to use rel_loss', action='store_true')

        # Arguments for visualization
        parser.add_argument('-prd_to_view', help='Specify a predicate list to view embeddings', type=str, nargs='+')
        parser.add_argument('-reduce_method', help='Method to reduce high-dimensional data', type=str, default='pca')
        parser.add_argument('-num_example_per_prd', help='Number of examples for each predicate', type=int, default=5)

        # Arguments for CrossAttGCN
        parser.add_argument('-use_obj', dest='use_obj', help='use obj module', action='store_true')
        parser.add_argument('-obj_time_step_num', dest='obj_time_step_num', help='time step number of obj', type=int, default=3)
        parser.add_argument('-obj_hidden_dim', dest='obj_hidden_dim', help='node hidden state dimension of obj', type=int, default=512)
        parser.add_argument('-obj_output_dim', dest='obj_output_dim', help='node output feature dimension of obj', type=int, default=512)
        parser.add_argument('-use_obj_knowledge', dest='use_obj_knowledge', help='use object cooccurrence knowledge', action='store_true')
        parser.add_argument('-obj_knowledge', dest='obj_knowledge', help='Filename to load matrix of object cooccurrence knowledge', type=str, default='')
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='node hidden state dimension', type=int, default=1024)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='pooling dimension', type=int, default=4096)

        parser.add_argument('-use_dualResGCN_rel', dest='use_dualResGCN_rel', help='use dualResGCN_rel module', action='store_true')
        parser.add_argument('-dualResGCN_rel_hidden_dim', dest='dualResGCN_rel_hidden_dim', help='node hidden state dimension of dualResGCN_rel', type=int, default=512)
        parser.add_argument('-dualResGCN_rel_output_dim', dest='dualResGCN_rel_output_dim', help='node output feature dimension of dualResGCN_rel', type=int, default=512)
        parser.add_argument('-use_rel_knowledge', dest='use_rel_knowledge', help='use cooccurrence knowledge of object pairs and relationships', action='store_true')
        parser.add_argument('-pred_weight', dest='pred_weight', action='store_true')

        parser.add_argument('-old_split_atten_map', help='use original Split_Atten_map codes, just for compatibility', action='store_true')
        parser.add_argument('-no_freq_gate', help='not to use frequency gate', action='store_true')
        parser.add_argument('-no_bias_in_training', help='not to use bias in training',  action='store_true')
        parser.add_argument('-no_bias', help='not to use bias at all',  action='store_true')
        parser.add_argument('-nms_thresh', help='threshold for NMS post-processing', type=float, default=0.5)

        # Arguments for KERN
        parser.add_argument('-ggnn_rel_time_step_num', dest='ggnn_rel_time_step_num', help='time step number of GGNN_rel', type=int, default=3)
        parser.add_argument('-ggnn_rel_hidden_dim', dest='ggnn_rel_hidden_dim',
                            help='node hidden state dimension of GGNN_rel', type=int, default=512)
        parser.add_argument('-ggnn_rel_output_dim', dest='ggnn_rel_output_dim',
                            help='node output feature dimension of GGNN_rel', type=int, default=512)
        parser.add_argument('-rel_knowledge', dest='rel_knowledge', help='Filename to load matrix of cooccurrence knowledge of object pairs and relationships',
                            type=str, default='prior_matrices/rel_matrix.npy')
        parser.add_argument('-test_split_size', help='Split size for batch in testing', type=int, default=1024)

        # Arguments for Motifs
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='leftright')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=2)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=4)
        parser.add_argument('-motifs_hidden_dim', help='node hidden state dimension', type=int, default=512)
        parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)

        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_bimodal_rel', dest='use_bimodal_rel',  action='store_true')

        # Arguments for VCTree
        parser.add_argument('-use_rl_tree', dest='use_rl_tree',  action='store_true')

        # Arguments for Bimodal
        parser.add_argument('-use_gaussian', dest='use_gaussian', help='use Gaussian embedding', action='store_true')
        parser.add_argument('-gaussian_reg', dest='gaussian_reg', help='type of regularization for Gaussian embedding', type=str, default='entropy')
        parser.add_argument('-uncer_margin', dest='uncer_margin', help='uncertainty margin used for regularization', type=float, default=200)
        parser.add_argument('-reg_weight', dest='reg_weight', help='weight for regularization', type=float, default=0.0001)
        parser.add_argument('-metric', dest='metric', help='Metric to compute match probability', type=str, default='w-distance')

        # Arguments for visual Gaussian
        parser.add_argument('-visual_gaussian', dest='visual_gaussian', help='use Gaussian embedding for only visual branch', action='store_true')
        parser.add_argument('-num_gaussian_samples', dest='num_gaussian_samples', help='number of Gaussian samples', type=int, default=8)
        parser.add_argument('-sampling_loss_weight', dest='sampling_loss_weight', help='weight for loss of sampling vector', type=float, default=0.1)
        parser.add_argument('-mu_plus_so_feat', dest='mu_plus_so_feat', help='add sbj/obj feature to original mu', action='store_true')
        parser.add_argument('-bias_to_z', help='add bias to rel_dists_z', action='store_true')

        parser.add_argument('-tb_log_dir', dest='tb_log_dir', help='dir to save tensorboard summaries', type=str, default='')

        return parser


cfg = ModelConfig()
