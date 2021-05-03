import torch


def distance_function(metric='COSIM'):
    if metric == 'COSIM':
        return torch.nn.CosineSimilarity
    elif metric == 'KLDiv':
        return torch.nn.KLDivLoss
    else:
        print('Undefined distance metric')


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

    assert(src_shape == trg_shape)

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
            src_to_trg_associations.append(trg_associated_pxl)
    return pixels_with_cycle_association


def gradient_diffusion(features):
    pass


def calc_contrastive_loss(final_pred_src, final_pred_trg):
    pixels_with_cycle_association = get_pixels_with_cycle_association(final_pred_src, final_pred_trg)
    loss_cass = 0

    for association in pixels_with_cycle_association:
        pass

    loss_cass *= -1/abs(len(pixels_with_cycle_association))

    return loss_cass
