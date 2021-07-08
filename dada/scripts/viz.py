import argparse
import torch
from torch.utils import data
from torch import nn
import numpy as np

from advent.dataset.cityscapes import CityscapesDataSet
from dada.dataset.synthia import SYNTHIADataSetDepth
from dada.domain_adaptation.config import cfg, cfg_from_file
from dada.model.deeplabv2 import get_deeplab_v2
from dada.domain_adaptation.contrastive_learning import distance_function
from dada.domain_adaptation.contrastive_learning import get_pixels_with_cycle_association


def create_mask(mask):
    # interpolate mask vector
    interp = nn.Upsample(
        size=(365, 365),
        mode="bilinear",
        align_corners=True,
    )

    # interpolated mask
    mask_scaled = interp(mask.view(1, 1, 46, 46))
    mask_scaled.shape

    # turn mask into np array
    mask_np = mask_scaled.numpy()
    mask_map = np.squeeze(mask_np)

    # Create another plotting mask
    masked = np.ma.masked_where(mask_map == 0, mask_map)
    return masked


def create_map(maski, maski_2, mask_j, img, img_trg, i_iter, flag=0):
    masked_i = create_mask(maski)
    masked_i_2 = create_mask(maski_2)
    masked_j = create_mask(mask_j)

    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    f, ax = plt.subplots(3, 2, figsize=(15, 15))

    # Grayscaling original to avoid discoloration issue
    img_gray = img[0, :, :]
    img_trg_gray = img_trg[0, :, :]

    # Source + i
    ax[0, 0].imshow(img_gray, cmap='gray')
    ax[0, 0].set_ylabel("original image")

    ax[0, 1].imshow(img_gray, cmap='gray')
    ax[0, 1].imshow(masked_i, interpolation='none')
    ax[0, 1].set_ylabel("original image + i pixels")

    # Source + i_2
    ax[1, 0].imshow(img_gray, cmap='gray')
    ax[1, 0].set_ylabel("original image")

    ax[1, 1].imshow(img_gray, cmap='gray')
    ax[1, 1].imshow(masked_i_2, interpolation='none')
    ax[1, 1].set_ylabel("original image + i* pixels")

    # Target + j

    ax[2, 0].imshow(img_trg_gray, cmap='gray')
    ax[2, 0].set_ylabel("original target image")

    ax[2, 1].imshow(img_trg_gray, cmap='gray')
    ax[2, 1].imshow(masked_j, interpolation='none')
    ax[2, 1].set_ylabel("original target image + j pixels")

    f.savefig("./img/maps/map_source_{}.jpg".format(i_iter))


def visualize_pixel_cycle_associations(model_path):
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

    model = get_deeplab_v2(
        num_classes=cfg.NUM_CLASSES,
        multi_level=cfg.TRAIN.MULTI_LEVEL
    )
    saved_state_dict = torch.load(model_path)

    start_iter = saved_state_dict['iter']
    model.load_state_dict(saved_state_dict['state_dict'])

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    _, batch = trainloader_iter.__next__()
    images_source, labels, _, _ = batch
    _, pred_src_main = model(images_source.cuda(device))

    _, batch = targetloader_iter.__next__()
    images, _, _, _ = batch
    _, pred_trg_main = model(images.cuda(device))

    dis_fn = distance_function(metric='COSIM')

    d1 = dis_fn(pred_src_main, pred_trg_main)
    d2 = dis_fn(pred_trg_main, pred_src_main)

    # get the pixels which have cycle association and mask vector
    pixels_with_cycle_association, mask_i, mask_i_2, mask_j = get_pixels_with_cycle_association(d1, d2, labels)

    mask_i = torch.zeros([1, 2116])
    mask_i2 = torch.zeros([1, 2116])
    mask_j = torch.zeros([1, 2116])

    for association in pixels_with_cycle_association:
        i, j, i2 = association
        mask_i[0, i] = 1
        mask_i2[0, i2] = 1
        mask_j[0, j] = 1

    i = 0
    create_map(mask_i, mask_i2, mask_j, images_source[0].numpy(), images[0].numpy(), i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='path of the model')
    parser.add_argument('--cfg',
                        help='path to config file')

    args = parser.parse_args()

    visualize_pixel_cycle_associations(args.model_path)
