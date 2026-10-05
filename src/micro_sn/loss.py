import cv2
import torch
import numpy as np


def downsample_mask(mask_tensor, size):
    mask = mask_tensor.numpy().squeeze()
    assert mask.ndim == 2, "Input mask tensor must be 2D"
    small = cv2.resize(mask.astype("uint8"), (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(small).long()

def ag_bce(pred, expert, student, hard_weight=4, eps=1e-6):
    assert pred.shape == expert.shape == student.shape, "Predictions and masks must have the same shape"
    expert = expert.float()
    hard = expert != student.float()
    weight = torch.ones_like(pred)
    weight[hard] = hard_weight
    bce = -(expert * torch.log(pred + eps) + (1.0 - expert) * torch.log(1.0 - pred + eps))
    return (weight * bce).mean()