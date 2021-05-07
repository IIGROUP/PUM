import os
from dataloaders.visual_genome import VG, vg_collate
from lib.utils import define_model, load_ckpt, do_test
from config import cfg
from torch.utils.data import DataLoader


test_data = VG(cfg.test_data_name, num_val_im=5000, filter_duplicate_rels=True,
               use_proposals=cfg.use_proposals, filter_non_overlap=cfg.mode == 'sgdet',
               num_im=cfg.num_im)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=cfg.num_gpus,
    shuffle=False,
    num_workers=cfg.num_workers,
    collate_fn=lambda x: vg_collate(x, mode='rel', num_gpus=cfg.num_gpus, is_train=True if cfg.test_data_name == 'train' else False),
    drop_last=True,
    pin_memory=True,
)

if cfg.cache is not None and os.path.exists(cfg.cache):
    # No need to load model
    detector = None
else:
    detector = define_model(cfg, test_data.ind_to_classes, test_data.ind_to_predicates)
    load_ckpt(detector, cfg.ckpt)
    detector.cuda()

do_test(detector, test_data, test_loader)
