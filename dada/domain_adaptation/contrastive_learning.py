import torch
import torch.nn.functional as F
import numpy as np


def distance_function(metric='COSIM'):
    if metric == 'COSIM':
        return torch.nn.CosineSimilarity(dim=0)
    elif metric == 'KLDiv':
        return torch.nn.KLDivLoss()
    else:
        print('Undefined distance metric')


def dis_fn_normalized(x, y, mean, std, dis_f):
    return (dis_f(x, y) - mean) / std


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

    returns a list of list containing tuples i, i*, j*
    """

    # the list contains all pixels which have cycle association
    # [[i, j*, i*], ....]
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

    src_to_trg_associations = torch.zeros((src_shape[0], src_shape[1], 2))
    trg_to_src_associations = torch.zeros((src_shape[0], src_shape[1], 2))

    # source to target associations
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            feature_src = features_source[i, j, :]
            max_sim = 0

            for k in range(trg_shape[0]):
                for l in range(trg_shape[1]):
                    feature_trg = features_target[k, l, :]
                    sim = distance_fn(feature_src, feature_trg)

                    if sim > max_sim:
                        src_to_trg_associations[i, j, 0] = k
                        src_to_trg_associations[i, j, 1] = l
                        max_sim = sim

    # target to source association
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            feature_src = features_target[i, j, :]
            max_sim = 0

            for k in range(trg_shape[0]):
                for l in range(trg_shape[1]):
                    feature_trg = features_source[k, l, :]
                    sim = distance_fn(feature_trg, feature_src)

                    if sim > max_sim:
                        trg_to_src_associations[i, j, 0] = k
                        trg_to_src_associations[i, j, 1] = l
                        max_sim = sim

    # check if source to target and target to source belong to same semantic class
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            x, y = src_to_trg_associations[i, j, :]
            x2, y2 = trg_to_src_associations[x, y, :]
            if torch.argmax(features_source[i, j, :]) == torch.argmax(features_source[i, j, :]):
                pixels_with_cycle_association.append([(i, j), (x, y), (x2, y2)])

    return pixels_with_cycle_association


def spatial_aggregation(features, alpha=0.5, metric='COSIM'):
    """
    gradient diffusion using spatial aggregation
    """
    dimX, dimY, dimF = list(features.shape)
    dis = distance_function(metric)

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
                    F_2 += weight * features[k, l, :]

            F_ = features[i, j, :]
            F_hat = (1 - alpha) * F_ + alpha * F_2
            features[i, j, :] = F_hat

    return features


def calc_contrastive_loss(final_pred_src, final_pred_trg):
    """
        calculate the contrastive association loss
    """
    # perform spatial aggregation on target before softmax and cycle association
    final_pred_trg = spatial_aggregation(final_pred_trg, alpha=0.5, metric='KLDiv')

    # perform softmax and get the probablities
    final_pred_src = F.softmax(final_pred_src, dim=2)
    final_pred_trg = F.softmax(final_pred_trg, dim=2)

    # get the pixels which have cycle association
    pixels_with_cycle_association = get_pixels_with_cycle_association(final_pred_src, final_pred_trg, 'KLDiv')
    dis = distance_function(metric='KLDiv')
    loss_cass = 0

    dimX, dimY, dimF = list(final_pred_src.shape)

    # calculate the contrastive association loss
    for association in pixels_with_cycle_association:
        i_x, i_y = association[0]
        j_x, j_y = association[1]
        i2_x, i2_y = association[2]

        num = (torch.exp(dis(final_pred_src[i_x, i_y, :], final_pred_trg[j_x, j_y, :])) *
               torch.exp(dis(final_pred_trg[j_x, j_y, :], final_pred_trg[i2_x, i2_y, :])))
        den1 = 0
        den2 = 0
        for i in range(dimX):
            for j in range(dimY):
                if i != j_x and j != j_y:
                    den1 += torch.exp(dis(final_pred_src[i_x, i_y, :], final_pred_trg[i, j, :]))
                if i != i2_x and j != i2_y:
                    den2 += torch.exp(dis(final_pred_src[j_x, j_y, :], final_pred_trg[i, j, :]))

        loss_cass += torch.log(num / (den1 * den2))

    loss_cass *= -1 / abs(len(pixels_with_cycle_association))

    return loss_cass
