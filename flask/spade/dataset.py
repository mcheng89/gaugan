"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torchvision.transforms as transforms
from PIL import Image

def __scale_width(img, target_width, method=Image.BICUBIC):
  ow, oh = img.size
  if (ow == target_width):
    return img
  w = target_width
  h = int(target_width * oh / ow)
  return img.resize((w, h), method)

def get_transform(opt, method=Image.BICUBIC, normalize=True):
  transform_list = []
  transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt['load_size'], method)))
  transform_list += [transforms.ToTensor()]
  if normalize:
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  
  return transforms.Compose(transform_list)
