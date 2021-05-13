# --------------------------------------------------------
# Domain adaptation training
# Copyright (c) 2019 valeo.ai
#
# Written by Rohit Kaushik
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.domain_adaptation.train_UDA import print_losses, log_losses_tensorboard
from advent.utils.func import adjust_learning_rate
from advent.utils.func import loss_calc

from dada.utils.viz_segmask import colorize_mask
from dada.domain_adaptation.contrastive_learning import calc_contrastive_loss


def train_dada(model, trainloader, targetloader, cfg):
    """ Contrastive training with dada
    """
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(
        model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
        lr=cfg.TRAIN.LEARNING_RATE,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    # interpolate output segmaps
    interp = nn.Upsample(
        size=(input_size_source[1], input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )
    # interp_target = nn.Upsample(
    #     size=(input_size_target[1], input_size_target[0]),
    #     mode="bilinear",
    #     align_corners=True,
    # )
    torch.autograd.set_detect_anomaly(True)
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        _, pred_src_main = model(images_source.cuda(device))
        pred_src_main_interp = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main_interp, labels, device)

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        _, pred_trg_main = model(images.cuda(device))
        # pred_trg_main_interp = interp_target(pred_trg_main)

        _, dimF, dimX, dimY = pred_src_main.shape
        loss_contrastive = calc_contrastive_loss(pred_src_main.reshape((dimF, dimX*dimY)),
                                                 pred_trg_main.reshape(dimF, dimX*dimY),
                                                 labels)

        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_CONTRASTIVE_MAIN * loss_contrastive)
        loss.backward()

        optimizer.step()

        current_losses = {
            "loss_seg_src_main": loss_seg_src_main,
            "loss_contrastive_src_main": loss_contrastive
        }
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f"model_{i_iter}.pth")
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, "T")
                draw_in_tensorboard(
                    writer, images_source, i_iter, pred_src_main, num_classes, "S"
                )


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f"Image - {type_}", grid_image, i_iter)

    softmax = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    mask = colorize_mask(num_classes, np.asarray(np.argmax(softmax, axis=2), dtype=np.uint8)).convert("RGB")
    grid_image = make_grid(torch.from_numpy(np.array(mask).transpose(2, 0, 1)),
                           3,
                           normalize=False,
                           range=(0, 255))
    writer.add_image(f"Prediction - {type_}", grid_image, i_iter)


def train_domain_adaptation_with_contrastive_loss(model, trainloader, targetloader, cfg):
    assert cfg.TRAIN.DA_METHOD in {"DADA"}, "Not yet supported DA method {}".format(cfg.TRAIN.DA_METHOD)
    train_dada(model, trainloader, targetloader, cfg)
