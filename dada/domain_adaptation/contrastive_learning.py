import torch
import torch.nn.functional as F
import numpy as np


def distance_function(metric='COSIM'):
    if metric == 'COSIM':
        return torch.nn.CosineSimilarity(dim=0)
    elif metric == 'KLDiv':
        return torch.nn.KLDivLoss
    else:
        print('Undefined distance metric')


def contrast_normalization_factors(src_feature, trg_feature):
    """
    calculate the mean and variance to normalize distance metric
    """
    Ht, Wt, _ = list(trg_feature.shape)
    Hs, Ws, _ = list(src_feature.shape)

    mean_src_to_trg = 1 / Ht * Wt * 5


def get_pixels_with_cycle_association(features_source, features_target, dis_metric='COSIM'):
    """
    find pixels that have cycle associations between them. by default uses distance
    metric as cosine similarity
    """

    # the list contains all pixels which have cycle association
    # [[(i_x, i_y), (j_x, j_y)], ....]
    pixels_with_cycle_association = []

    src_shape = list(features_source.shape)
    trg_shape = list(features_target.shape)

    assert (src_shape == trg_shape)

    # the distance function that calculates similarity between pixel
    # i.e cosine similarity or KL divergence loss
    distance_fn = distance_function(dis_metric)

    # first find the association from source to target
    # (i_x, i_y) ---> (j_x, j_y)
    # then we check for association from trg to src
    # (j_x, j_y) ---> (i*_x, i*_y)

    src_to_trg_associations = []

    for i in src_shape[0]:
        for j in src_shape[1]:
            feature_src = features_source[i, j, :]
            max_sim = 0
            trg_associated_pxl = [-1, -1]

            for k in trg_shape[0]:
                for l in trg_shape[1]:
                    feature_trg = features_target[k, l, :]
                    sim = distance_fn(feature_src, feature_trg)

                    if sim > max_sim:
                        trg_associated_pxl = [k, l]
                        max_sim = sim
            src_to_trg_associations.append([(i, j), (trg_associated_pxl[0], trg_associated_pxl[1])])

    for src_trg_assoc in src_to_trg_associations:
        i_x, i_y = src_trg_assoc[0]
        j_x, j_y = src_trg_assoc[1]

        for i in src_shape[0]:
            for j in src_shape[1]:
                pass

    return pixels_with_cycle_association


def spatial_aggregation(features, alpha=0.5):
    """
    gradient diffusion using spatial aggregation
    """
    dimX, dimY, dimF = list(features.shape)
    dis = distance_function('COSIM')

    for i in range(dimX):
        for j in range(dimY):
            weight_denom = 0

            for k in range(dimX):
                for l in range(dimY):
                    if k == i and j == l:
                        continue
                    weight_denom += torch.exp(dis(features[i, j, :], features[k, l, :]))

            F_2 = 0
            for k in range(dimX):
                for l in range(dimY):
                    if k == i and j == l:
                        continue
                    weight = torch.exp(dis(features[i, j, :], features[k, l, :])) * weight_denom
                    F_2 += weight*features[k, l, :]

            F_ = features[i, j, :]
            F_hat = (1 - alpha) * F_ + alpha * F_2
            features[i, j, :] = F_hat


def calc_contrastive_loss(final_pred_src, final_pred_trg):
    pixels_with_cycle_association = get_pixels_with_cycle_association(final_pred_src, final_pred_trg)
    loss_cass = 0

    for association in pixels_with_cycle_association:
        pass

    loss_cass *= -1 / abs(len(pixels_with_cycle_association))

    return loss_cass
