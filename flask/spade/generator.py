"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spade.normalizer import SPADE

class SPADEGenerator(nn.Module):
  def __init__(self, opt):
    super().__init__()

    # nf: # of gen filters in first conv layer
    nf = 64

    self.sw, self.sh = self.compute_latent_vector_size(opt['crop_size'], opt['aspect_ratio'])

    self.fc = nn.Conv2d(opt['label_nc'], 16 * nf, 3, padding=1)

    self.head_0 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)

    self.G_middle_0 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)
    self.G_middle_1 = SPADEResnetBlock(opt, 16 * nf, 16 * nf)

    self.up_0 = SPADEResnetBlock(opt, 16 * nf, 8 * nf)
    self.up_1 = SPADEResnetBlock(opt, 8 * nf, 4 * nf)
    self.up_2 = SPADEResnetBlock(opt, 4 * nf, 2 * nf)
    self.up_3 = SPADEResnetBlock(opt, 2 * nf, 1 * nf)

    self.conv_img = nn.Conv2d(1 * nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)
  
  def compute_latent_vector_size(self, crop_size, aspect_ratio):
    num_up_layers = 5

    sw = crop_size // (2**num_up_layers)
    sh = round(sw / aspect_ratio)

    return sw, sh
  
  def forward(self, seg):
    # we downsample segmap and run convolution
    x = F.interpolate(seg, size=(self.sh, self.sw))
    x = self.fc(x)

    x = self.head_0(x, seg)

    x = self.up(x)
    x = self.G_middle_0(x, seg)
    x = self.G_middle_1(x, seg)

    x = self.up(x)
    x = self.up_0(x, seg)
    x = self.up(x)
    x = self.up_1(x, seg)
    x = self.up(x)
    x = self.up_2(x, seg)
    x = self.up(x)
    x = self.up_3(x, seg)

    x = self.conv_img(F.leaky_relu(x, 2e-1))
    x = torch.tanh(x)

    return x

import torch.nn.utils.spectral_norm as spectral_norm

# label_nc: the #channels of the input semantic map, hence the input dim of SPADE
# label_nc: also equivalent to the # of input label classes
class SPADEResnetBlock(nn.Module):
  def __init__(self, opt, fin, fout):
    super().__init__()

    self.learned_shortcut = (fin != fout)
    fmiddle = min(fin, fout)

    self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
    self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
    if self.learned_shortcut:
      self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

    # define normalization layers
    self.norm_0 = SPADE(opt, fin)
    self.norm_1 = SPADE(opt, fmiddle)
    if self.learned_shortcut:
      self.norm_s = SPADE(opt, fin)

  # note the resnet block with SPADE also takes in |seg|,
  # the semantic segmentation map as input
  def forward(self, x, seg):
    x_s = self.shortcut(x, seg)

    dx = self.conv_0(self.relu(self.norm_0(x, seg)))
    dx = self.conv_1(self.relu(self.norm_1(dx, seg)))

    out = x_s + dx
    return out

  def shortcut(self, x, seg):
    if self.learned_shortcut:
      x_s = self.conv_s(self.norm_s(x, seg))
    else:
      x_s = x
    return x_s

  def relu(self, x):
    return F.leaky_relu(x, 2e-1)


