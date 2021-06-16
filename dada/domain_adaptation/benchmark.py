import time
import sys

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from dada.domain_adaptation import contrastive_learning


def benchmark_contrastive_learning(model, trainloader, targetloader, cfg):
    """ Contrastive training with dada
    """
    # Create the model and start the training.
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    # SEGMNETATION NETWORK
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    torch.autograd.set_detect_anomaly(True)
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _, _ = batch
        _, pred_src_main, _ = model(images_source.cuda(device))

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        _, pred_trg_main, _ = model(images.cuda(device))

        _, dimF, dimX, dimY = pred_src_main.shape

        final_pred_trg = pred_trg_main.reshape(dimF, dimX*dimY).t()
        final_pred_src = pred_src_main.reshape((dimF, dimX*dimY)).t()

        # perform spatial aggregation on target before softmax and cycle association
        tic = time.perf_counter()
        final_pred_trg = contrastive_learning.spatial_aggregation(final_pred_trg, alpha=0.5)
        toc = time.perf_counter()
        print(f"Spatial Aggregation takes {toc - tic:0.4f} seconds")
        assert (final_pred_trg.shape == final_pred_src.shape)

        cosine_dis = contrastive_learning.distance_function(metric='COSIM')

        # calculate the pixel wise distance
        tic = time.perf_counter()
        d1 = cosine_dis(final_pred_src, final_pred_trg)
        toc = time.perf_counter()
        print(f"Pixel wise distance from source to target takes {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        d2 = cosine_dis(final_pred_trg, final_pred_src)
        toc = time.perf_counter()
        print(f"Pixel wise distance from target to source takes {toc - tic:0.4f} seconds")

        # get the pixels which have cycle association
        tic = time.perf_counter()
        pixels_with_cycle_association = contrastive_learning.get_pixels_with_cycle_association(d1, d2, labels)
        toc = time.perf_counter()
        print(f"Cycle association takes {toc - tic:0.4f} seconds")

        # contrast normalize the distance values
        tic = time.perf_counter()
        u, sig = contrastive_learning.contrast_normalization_factors(d1)
        toc = time.perf_counter()
        print(f"Calculating normalization factor for source to target distance takes {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        u2, sig2 = contrastive_learning.contrast_normalization_factors(d2)
        toc = time.perf_counter()
        print(f"Calculating normalization factor for target to source distance takes {toc - tic:0.4f} seconds")

        d1 = (d1 - u) / sig
        d2 = (d2 - u2) / sig2

        tic = time.perf_counter()

        d1softmax = torch.nn.functional.softmax(d1, dim=1)
        d2softmax = torch.nn.functional.softmax(d2, dim=1)

        I = []
        J = []
        I_2 = []

        for association in pixels_with_cycle_association:
            i, j, i_2 = association
            I.append(i)
            J.append(j)
            I_2.append(i_2)

        loss = torch.sum(torch.log(d1softmax[I, J] * d2softmax[J, I_2]))

        loss *= -1 / abs(len(pixels_with_cycle_association))

        toc = time.perf_counter()
        print(f"Lfass loss calculation takes {toc - tic:0.4f} seconds")

        sys.stdout.flush()
