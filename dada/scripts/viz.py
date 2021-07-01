import argparse
import torch
from torch.utils import data
import numpy as np

from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

from dada.model.deeplabv2_depth import get_deeplab_v2_depth
from dada.domain_adaptation.config import cfg, cfg_from_file
from dada.dataset.synthia import SYNTHIADataSetDepth
from advent.dataset.cityscapes import CityscapesDataSet


def grad_cam_visualization(model_path, cls):
    model = get_deeplab_v2_depth(
        num_classes=cfg.NUM_CLASSES,
        multi_level=cfg.TRAIN.MULTI_LEVEL
    )

    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict['state_dict'])

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

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

    targetloader_iter = enumerate(target_loader)

    _, batch = targetloader_iter.__next__()
    image, _, _, _ = batch

    clsroi = ClassRoI(model=model, image=None, cls=cls)
    newsgc = SegGradCAM(model, image, cls, roi=clsroi,
                        normalize=True, abs_w=False, posit_w=False)
    newsgc.SGC()

    plotter = SegGradCAMplot(newsgc, model=model, n_classes=cfg.NUM_CLASSES, outfolder='./viz')
    plotter.explainPixel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='path of the model')

    args = parser.parse_args()

    grad_cam_visualization(args.model_path, 7)
