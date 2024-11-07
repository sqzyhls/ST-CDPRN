import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds


def set_requires_grad(module: nn.Module, requires_grad: bool):
    print("In model utils: set_requires_grad()module requires_grad")
    for p in module.parameters():
        p.requires_grad_(requires_grad)



def default(x, d):
    print("In model utils: default()x d")
    return d if x is None else x


def get_num_points(x: Pointclouds, /):
    print("In model utils: get_num_points()x /")
    return x.points_padded().shape[1]


def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    print("In model utils: get_custom_betas()beta_start beta_end warmup_frac num_train_timesteps")
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas
