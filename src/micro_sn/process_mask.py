import cv2
import torch
import numpy as np


def downsample_mask(mask_tensor, size):
    mask = mask_tensor.numpy().squeeze()
    assert mask.ndim == 2, "Input mask tensor must be 2D"
    small = cv2.resize(mask.astype("uint8"), (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(small).long()
