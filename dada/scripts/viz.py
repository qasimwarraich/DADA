import argparse
import torch
from torch.utils import data
import numpy as np

from advent.dataset.cityscapes import CityscapesDataSet
from dada.dataset.synthia import SYNTHIADataSetDepth
from dada.domain_adaptation.config import cfg, cfg_from_file
from advent.model.deeplabv2 import get_deeplab_v2
from dada.domain_adaptation.contrastive_learning import distance_function

device = torch.device('gpu')


def visualize_pixel_cycle_associations(model_path):
    cfg_from_file(args.cfg)

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    # DATALOADERS
    source_dataset = SYNTHIADataSetDepth(
        root=cfg.DATA_DIRECTORY_SOURCE,
        list_path=cfg.DATA_LIST_SOURCE,
        set=cfg.TRAIN.SET_SOURCE,
        num_classes=cfg.NUM_CLASSES,
        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
        crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
        mean=cfg.TRAIN.IMG_MEAN,
        use_depth=cfg.USE_DEPTH,
    )
    source_loader = data.DataLoader(
        source_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    target_dataset = CityscapesDataSet(
        root=cfg.DATA_DIRECTORY_TARGET,
        list_path=cfg.DATA_LIST_TARGET,
        set=cfg.TRAIN.SET_TARGET,
        info_path=cfg.TRAIN.INFO_TARGET,
        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
        crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
        mean=cfg.TRAIN.IMG_MEAN
    )

    target_loader = data.DataLoader(
        target_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )

    model = get_deeplab_v2(
        num_classes=cfg.NUM_CLASSES,
        multi_level=cfg.TRAIN.MULTI_LEVEL
    )
    saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)

    start_iter = saved_state_dict['iter']
    model.load_state_dict(saved_state_dict['state_dict'])

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    _, batch = trainloader_iter.__next__()
    images_source, labels, _, _ = batch

    dis_fn = distance_function(metric='COSIM')

    _, pred_src_main = model(images_source.cuda(device))

    d1 = dis_fn(src_feature, trg_feature)
    d2 = dis_fn(trg_feature, src_feature)

    # get the pixels which have cycle association and mask vector
    pixels_with_cycle_association, mask_i, mask_i_2, mask_j = get_pixels_with_cycle_association(d1, d2, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='path of the model')

    args = parser.parse_args()

    visualize_pixel_cycle_associations(args.model_path, 7)
