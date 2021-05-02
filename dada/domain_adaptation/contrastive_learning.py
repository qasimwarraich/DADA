import torch


def get_pixels_with_cycle_association(features_source, features_target, distance_metric='COSIM'):
    """
    find pixels that have cycle associations between them. by default uses distance
    metric as cosine similarity
    """
    pixels = []
    return pixels


def gradient_diffusion(features):
    pass


def calc_contrastive_loss(final_pred_src, final_pred_trg):
    pixels_with_cycle_association = get_pixels_with_cycle_association(final_pred_src, final_pred_trg)
    loss_cass = 0

    for i, j in pixels_with_cycle_association:
        pass

    loss_cass *= -1/abs(len(pixels_with_cycle_association))

    return loss_cass
