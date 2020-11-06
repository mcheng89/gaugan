"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d

# norm_nc: the #channels of the normalized activations, hence the output dim of SPADE
# label_nc: the #channels of the input semantic map, hence the input dim of SPADE
# label_nc: also equivalent to the # of input label classes
class SPADE(nn.Module):
  def __init__(self, opt, norm_nc):
    super().__init__()

    self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)

    # number of internal filters for generating scale/bias
    nhidden = 128
    # size of kernels
    kernal_size = 3
    # padding size
    padding = kernal_size // 2

    self.mlp_shared = nn.Sequential(
      nn.Conv2d(opt['label_nc'], nhidden, kernel_size=kernal_size, padding=padding),
      nn.ReLU()
    )
    self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernal_size, padding=padding)
    self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernal_size, padding=padding)

  def forward(self, x, segmap):
    # Part 1. generate parameter-free normalized activations
    normalized = self.param_free_norm(x)

    # Part 2. produce scaling and bias conditioned on semantic map
    # resize input segmentation map to match x.size() using nearest interpolation
    # N, C, H, W = x.size()
    segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
    actv = self.mlp_shared(segmap)
    gamma = self.mlp_gamma(actv)
    beta = self.mlp_beta(actv)

    # apply scale and bias
    out = normalized * (1 + gamma) + beta

    return out
