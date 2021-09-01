import torch 

def get_pixels_with_cycle_association(dis_src_to_trg, dis_trg_to_src, labels, threshold=None, cls=None):
    """
    @dis_src_to_trg (tensor): n * n, distance between every pair of pixel from source to target
    @dis_trg_to_src (tensor): n * n, distance between every pair of pixel from target to source
    @label (tensor): the true class labels for source pixels
    @cls (integer): only return associations belonging to this class, useful in visualization

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

    dimX, dimY = dis_src_to_trg.shape

    new_labels = labels
    new_labels = new_labels.reshape(dimX)

    print("\nsrc to trg:\n", dis_src_to_trg)
    print("\nsrc labels:\n", new_labels)
    print("\ntrg to src:\n",dis_trg_to_src)
    closest_pixels_in_trg = torch.argmax(dis_src_to_trg, dim=1)
    closest_pixels_in_src = torch.argmax(dis_trg_to_src, dim=1)
    
    print("\nclosest pixels in src and trg respectively:")
    print("src: ", closest_pixels_in_src)
    print("trg: ", closest_pixels_in_trg)

    for i in range(dimX): #Maybe should be X*Y to check all labels as the labels are flattened?
        j = closest_pixels_in_trg[i].item()
        i_2 = closest_pixels_in_src[j].item()
        print("\ni, j, i_2:")
        print(i , j , i_2, sep = ', ')


        if threshold is not None and not (dis_src_to_trg[i, j].item() > threshold):
            continue

        if new_labels[i].item() == new_labels[i_2].item():
            print("\nmatched label:", new_labels[i].item())
            if cls is not None and not (new_labels[i].item() == cls):
                continue
            pixels_with_cycle_association.append([i, j, i_2])

    print("\npca array", pixels_with_cycle_association)
    return pixels_with_cycle_association
