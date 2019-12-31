import os
import sys
import torch
import numpy as np
import torchvision.utils as vision_utils

def make_image_summary(images_to_show, normalize):
    image_slices_to_show = []
    grids = {}

    axis = 0
    slice_idx = images_to_show[0].size()[axis+1] // 2
    image_slices = []
    for image in images_to_show:
        image_slice = torch.select(image, axis+1, slice_idx)
        image_slices.append(image_slice)
    image_slices_to_show += image_slices
    grids['images'] = vision_utils.make_grid(image_slices_to_show, pad_value=1, nrow=len(images_to_show),
                                                 normalize=normalize, range=None)
    return grids