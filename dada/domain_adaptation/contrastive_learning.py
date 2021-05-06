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

    dimF, dimX, dimY = src_shape

    for i in range(dimX):
        for j in range(dimY):
            # source to target associations
            feature_src = features_source[:, i, j]
            d = distance_fn(torch.reshape(feature_src, (16, 1, 1)), features_target)
            d[i, j] = 0
            src_to_trg_associations[i, j] = (d == torch.max(d)).nonzero()

            # trg to src associations
            feature_trg = features_target[:, i, j]
            d = distance_fn(torch.reshape(feature_trg, (16, 1, 1)), features_source)
            d[i, j] = 0
            trg_to_src_associations[i, j] = (d == torch.max(d)).nonzero()

    # check if source to target and target to source belong to same semantic class
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            x, y = list(src_to_trg_associations[i, j, :])
            x2, y2 = list(trg_to_src_associations[x, y, :])
            if torch.argmax(features_source[:, i, j]) == torch.argmax(features_source[:, x2, y2]):
                pixels_with_cycle_association.append([(i, j), (x, y), (x2, y2)])

    return pixels_with_cycle_association


def spatial_aggregation(features, alpha=0.5, metric='COSIM'):
    """
    gradient diffusion using spatial aggregation
    """
    dimF, dimX, dimY = list(features.shape)
    dis = distance_function(metric)

    for i in range(dimX):
        for j in range(dimY):
            d = dis(torch.reshape(features[:, i, j], (16, 1, 1)), features)
            assert(list(d.shape) == [dimX, dimY])

            # set distance to itself (same pixel) as 0
            d[i, j] = 0

            d_exp = torch.exp(d)
            d_exp[i, j] = 0

            weight_denom = torch.sum(d_exp)
            weight = d_exp * weight_denom
            weight[i, j] = 0

            F_2 = torch.sum(weight*features)
            F_ = features[:, i, j]

            assert(F_2.shape == F_.shape)

            F_hat = (1 - alpha) * F_ + alpha * F_2
            features[:, i, j] = F_hat

    return features


def calc_contrastive_loss(final_pred_src, final_pred_trg):
    """
        calculate the contrastive association loss
    """
    # perform spatial aggregation on target before softmax and cycle association
    final_pred_trg = spatial_aggregation(final_pred_trg, alpha=0.5)

    # perform softmax and get the probablities
    final_pred_src = F.softmax(final_pred_src, dim=0)
    final_pred_trg = F.softmax(final_pred_trg, dim=0)

    # get the pixels which have cycle association
    pixels_with_cycle_association = get_pixels_with_cycle_association(final_pred_src, final_pred_trg, 'KLDiv')
    dis = distance_function(metric='KLDiv')
    loss_cass = 0

    dimF, dimX, dimY = list(final_pred_src.shape)

    # calculate the contrastive association loss
    for association in pixels_with_cycle_association:
        i_x, i_y = association[0]
        j_x, j_y = association[1]
        i2_x, i2_y = association[2]

        num = (torch.exp(dis(final_pred_src[i_x, i_y, :], final_pred_trg[j_x, j_y, :])) *
               torch.exp(dis(final_pred_trg[j_x, j_y, :], final_pred_trg[i2_x, i2_y, :])))

        d1 = dis(final_pred_src[:, i_x, i_y], final_pred_trg)
        d1[j_x, j_y] = 0
        den1 = torch.sum(torch.exp(d1))

        d2 = dis(final_pred_src[:, j_x, j_y], final_pred_trg)
        d2[i2_x, i2_y] = 0
        den2 = torch.sum(torch.exp(d2))

        loss_cass += torch.log(num / (den1 * den2))

    loss_cass *= -1 / abs(len(pixels_with_cycle_association))

    return loss_cass
