import argparse
import torch
from torch.utils import data
from torch import nn
import numpy as np
import sys

from advent.dataset.cityscapes import CityscapesDataSet
from dada.dataset.synthia import SYNTHIADataSetDepth
from dada.domain_adaptation.config import cfg, cfg_from_file
from dada.model.deeplabv2_resnet18 import get_deeplab_v2_resnet18
from dada.domain_adaptation.contrastive_learning import distance_function
from dada.domain_adaptation.contrastive_learning import get_pixels_with_cycle_association
from tqdm import tqdm


def oracle():
    cfg_from_file(args.cfg)
    device = cfg.GPU_ID

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

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    false_examples = 0
    class_acc = {i: 0 for i in range(16)}
    class_acc[255] = 0

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch

        _, batch = targetloader_iter.__next__()
        images, labels_trg, _, _ = batch

        unique_classes_src = torch.unique(labels)
        unique_classes_trg = torch.unique(labels_trg)

        flag = False
        for i, x in enumerate(unique_classes_src):
            if x in unique_classes_trg:
                continue
            else:
                flag = True
                class_acc[x.item()] += 1
        if flag:
            false_examples += 1

    print('\nfalse examples out of {} samples are: {}\n'.format(cfg.TRAIN.EARLY_STOP, false_examples))
    print(class_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='path to config file')

    args = parser.parse_args()

    oracle()
