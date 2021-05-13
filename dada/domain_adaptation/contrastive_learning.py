import torch
import torch.nn.functional as F
from torch import nn
from torch import linalg as LA


def kl_div(p1, p2, dim=2):
    """
    @p1 (tensor): n * m where n is number of class and m is number of pixels
    @p2 (tensor): n * m where n is the number of class and m is number of pixels

    Computes a pairwise KL Divergence loss for each pair of pixel in p1 and p2

    returns (tensor): m * m tensor which (i, j) value is the KL divergence distance of pixel i with pixel j
    """
    assert(p1.shape == p2.shape)
    z = torch.log(torch.transpose(p1, 0, 1)).unsqueeze(1) - torch.log(torch.transpose(p2, 0, 1))
    return -1*torch.sum(torch.mul(torch.transpose(p1, 0, 1).unsqueeze(0), z), dim=dim)


def cosine_sim(x, y):
    """
    @x (tensor): n * m where n is number of class and m is number of pixels
    @y (tensor): n * m where n is the number of class and m is number of pixels

    Computes a pairwise Cosine Similarity for each pair of pixel in x and y

    returns (tensor): m * m tensor which (i, j) value is the cosine similarity of pixel i with pixel j
    """

    dimX, dimY = list(x.shape)

    assert(x.shape == y.shape)

    x_norm = LA.norm(x, 2, dim=0)
    y_norm = LA.norm(y, 2, dim=0)
    x1 = x / x_norm
    y1 = y / y_norm

    x1 = torch.reshape(x1, (dimX, dimY, 1))
    y1 = torch.reshape(y1, (dimX, 1, dimY))

    cosim = torch.bmm(x1, y1)
    cosim = torch.sum(cosim, dim=0)

    return cosim


def distance_function(metric='COSIM'):
    """
    @metric (str): name of the distance function

    returns (func): corresponding distance function
    """

    if metric == 'COSIM':
        return cosine_sim
    elif metric == 'KLDiv':
        return kl_div
    else:
        print('Undefined distance metric')


def contrast_normalization_factors(dis):
    """
    @dis (tensor): m * m tensor which contains the distance values for pair of pixels

    Computes mean and standard deviation for each pixel distance values with every other value. It is used
    to do contrast normalization on the distance values

    returns:
        @mean (tensor): m * 1 tensor, the mean value of similarities for each of m pixels
        @std  (tensor): m * 1 tensor, the standard deviation of similarities for each of m pixels
    """

    dimX, dimY = list(dis.shape)

    mean = (1 / dimX) * (torch.sum(dis, dim=1) - torch.diag(dis))

    std = torch.sqrt((1/(dimX - 1)) * torch.sum(torch.square(dis - mean)) - torch.square(torch.diag(dis) - mean))

    return mean, std


def get_pixels_with_cycle_association(dis_src_to_trg, dis_trg_to_src, labels):
    """
    @dis_src_to_trg (tensor): n * n, distance between every pair of pixel from source to target
    @dis_trg_to_src (tensor): n * n, distance between every pair of pixel from target to source
    @label (tensor): the true class labels for source pixels

    find pixels that have cycle associations between them. by default uses distance
    metric as cosine similarity

    returns:
        @pixels_with_cycle_association (list of tuples): tuple of pixel which have cycle consistency
            i.e one element of the list can be (i, j, i*)
            i  --> is any pixel in source
            j  --> is a pixel in target with which i has maximum similarity
            i* --> is a pixel in source with which j has maximum similarity
    """

    # the list contains all pixels which have cycle association
    # [[i, j*, i*], ....]
    pixels_with_cycle_association = []

    assert (dis_src_to_trg.shape == dis_trg_to_src.shape)

    interp_target = nn.Upsample(
        size=(46, 46),
        mode="bilinear",
        align_corners=True,
    )

    dimX, dimY = dis_src_to_trg.shape

    new_labels = interp_target(labels.view(1, 1, 365, 365))
    new_labels = new_labels.reshape(dimX)

    closest_pixels_in_trg = torch.argmax(dis_src_to_trg, dim=1)
    closest_pixels_in_src = torch.argmax(dis_trg_to_src, dim=1)

    for i in range(dimX):
        j = closest_pixels_in_trg[i].item()
        i_2 = closest_pixels_in_src[j].item()

        if new_labels[i] == new_labels[i_2]:
            pixels_with_cycle_association.append([i, j, i_2])

    return pixels_with_cycle_association


def spatial_aggregation(features, alpha=0.5, metric='COSIM'):
    """
    @features (tensor): c*n where c is number of class and n is number of pixel
    @alpha (float): constant that controls the ratio of aggregated features
    @metric (str): the name of distance function to use

    gradient diffusion using spatial aggregation

    returns:
        @features (tensor): spatial aggregated tensor
    """

    dimX, dimY = list(features.shape)
    dis = distance_function(metric)

    d = dis(features, features)
    u, sig = contrast_normalization_factors(d)
    d = (d - u) / sig
    d = torch.exp(d)

    weight_denom = torch.sum(d, dim=1) - torch.diag(d)

    weight = d / weight_denom

    weight.fill_diagonal_(0)

    features = (1-alpha)*features + alpha*torch.transpose(torch.mm(weight, torch.transpose(features, 0, 1)), 0, 1)

    return features


def calc_association_loss(src_feature, trg_feature, labels, dis_fn):
    """
    @src_feature (tensor): c*n where c is the number of class and n is number of pixel
    @trg_feature (tensor): c*n where c is the number of class and n is number of pixel
    @labels (tensor): class labels for source feature
    @dis_fn (func): distance function to use to establish cycle associations

    returns:
        @loss (float): the association loss
    """

    loss = 0

    # calculate the pixel wise distance
    d1 = dis_fn(src_feature, trg_feature)
    d2 = dis_fn(src_feature, trg_feature)

    # contrast normalize the distance values
    u, sig = contrast_normalization_factors(d1)
    u2, sig2 = contrast_normalization_factors(d2)

    d1 = (d1 - u) / sig
    d2 = (d2 - u2) / sig2

    # get the pixels which have cycle association
    pixels_with_cycle_association = get_pixels_with_cycle_association(d1, d2, labels)

    d1 = torch.exp(d1)
    d2 = torch.exp(d2)

    # calculate the contrastive association loss
    for association in pixels_with_cycle_association:
        i, j, i2 = association

        num = d1[i, j] * d2[j, i2]

        den1 = torch.sum(d1[i, :]) - d1[i, j]
        den2 = torch.sum(d2[j, :]) - d2[j, i2]

        loss += torch.log(num / (den1 * den2))

    loss *= -1 / abs(len(pixels_with_cycle_association))

    return loss


def calc_label_smooth_regularization(src_feature, trg_feature):
    """
    @src_feature (tensor):
    @trg_feature (tensor):
    """
    pass


def calc_contrastive_loss(final_pred_src, final_pred_trg, labels):
    """
    @final_pred_src (tensor): c*n, the final prediction feature for source
    @final_pred_trg (tensor): c*n, the final prediction feature for target
    @labels (tensor): the true label for source


    returns:
        @loss (float): the association loss
            consists of lcass and lfass
            lcass: association loss on final prediction probabilities
            lfass: association loss on final feature prediction
    """
    
    # perform spatial aggregation on target before softmax and cycle association
    final_pred_trg = spatial_aggregation(final_pred_trg, alpha=0.5)
    assert(final_pred_trg.shape == final_pred_src.shape)

    # cosine_dis = distance_function(metric='COSIM')
    # loss_fass = calc_association_loss(final_pred_src, final_pred_trg, labels, cosine_dis)

    # perform softmax and get the probablities
    final_pred_src = F.softmax(final_pred_src, dim=0)
    final_pred_trg = F.softmax(final_pred_trg, dim=0)

    kl_div_dis = distance_function(metric='KLDiv')
    loss_cass = calc_association_loss(final_pred_src, final_pred_trg, labels, kl_div_dis)

    return loss_cass
