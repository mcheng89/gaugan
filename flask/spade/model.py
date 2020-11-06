"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch
import torch.nn as nn
from torch.nn import init

from spade.generator import SPADEGenerator

class Pix2PixModel(torch.nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    self.FloatTensor = torch.cuda.FloatTensor if opt['use_gpu'] \
      else torch.FloatTensor

    self.netG = self.initialize_networks(opt)
  
  def forward(self, data, mode):
    input_semantics, real_image = self.preprocess_input(data)

    if mode == 'inference':
      with torch.no_grad():
        fake_image = self.generate_fake(input_semantics)
      return fake_image
    else:
      raise ValueError("|mode| is invalid")
  
  def preprocess_input(self, data):
    data['label'] = data['label'].long()

    # move to GPU and change data types
    if self.opt['use_gpu']:
      data['label'] = data['label'].cuda()
      data['instance'] = data['instance'].cuda()
      data['image'] = data['image'].cuda()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = self.FloatTensor(bs, self.opt['label_nc'], h, w).zero_()
    # one whole label map -> to one label map per class
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    return input_semantics, data['image']
  
  def generate_fake(self, input_semantics):
    fake_image = self.netG(input_semantics)
    return fake_image
  
  def create_network(self, cls, opt):
    net = cls(opt)
    if self.opt['use_gpu']:
      net.cuda()

    gain=0.02
    def init_weights(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
          init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
          init.constant_(m.bias.data, 0.0)
    # Applies fn recursively to every submodule (as returned by .children()) as well as self
    net.apply(init_weights)
    
    return net
  
  def load_network(self, net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt['checkpoints_dir'], save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net

  def initialize_networks(self, opt):
    netG = self.create_network(SPADEGenerator, opt)

    if not opt['isTrain']:
      netG = self.load_network(netG, 'G', opt['which_epoch'], opt)
    
    # self.print_network(netG)

    return netG
  
  def print_network(self, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          % (type(net).__name__, num_params / 1000000))
    print(net)
  