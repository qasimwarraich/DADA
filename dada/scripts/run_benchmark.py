# --------------------------------------------------------
# DADA training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings

import numpy as np
import yaml
import torch
from torch.utils import data

from advent.scripts.train import get_arguments
from advent.dataset.cityscapes import CityscapesDataSet
from advent.model.deeplabv2 import get_deeplab_v2

from dada.dataset.synthia import SYNTHIADataSetDepth
from dada.model.deeplabv2_depth import get_deeplab_v2_depth
from dada.domain_adaptation.config import cfg, cfg_from_file

from dada.domain_adaptation import benchmark

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def main():
    # LOAD ARGS
    args = get_arguments()
    print("Called with args:")
    print(args)

    assert args.cfg is not None, "Missing cfg file"
    cfg_from_file(args.cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    start_iter = 0
    # LOAD SEGMENTATION NET
    assert osp.exists(
        cfg.TRAIN.RESTORE_FROM
    ), f"Missing init model {cfg.TRAIN.RESTORE_FROM}"
    if cfg.TRAIN.MODEL == "DeepLabv2_depth":
        model = get_deeplab_v2_depth(
            num_classes=cfg.NUM_CLASSES,
            multi_level=cfg.TRAIN.MULTI_LEVEL
        )
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if "DeepLab_resnet_pretrained_imagenet" in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split(".")
                if not i_parts[1] == "layer5":
                    new_params[".".join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            start_iter = saved_state_dict['iter']
            model.load_state_dict(saved_state_dict['state_dict'])
    elif cfg.TRAIN.MODEL == "DeepLabv2":
        model = get_deeplab_v2(
            num_classes=cfg.NUM_CLASSES,
            multi_level=cfg.TRAIN.MULTI_LEVEL
        )
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if "DeepLab_resnet_pretrained_imagenet" in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split(".")
                if not i_parts[1] == "layer5":
                    new_params[".".join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            start_iter = saved_state_dict['iter']
            model.load_state_dict(saved_state_dict['state_dict'])
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print("Model loaded")

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

    benchmark.benchmark_contrastive_learning(model, source_loader, target_loader, cfg)


if __name__ == "__main__":
    main()
