# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import numpy as np
import kornia
import kornia as K
from kornia.constants import Resample, BorderType, pi
from typing import Tuple, List, Union, Dict, cast, Optional
# def DiffAugment(x, policy = 'translation,cutout', channels_first=True):
# def DiffAugment(x, policy = 'color,translation,cutout', channels_first=True):


def DiffAugment(x, policy='translation', channels_first=True, DiffAugment=True, vis=False, isTrue=False):
    if (np.random.random() < 0.15) and (vis==False):
        rgb = F.interpolate(x[:,:3], size=(128, 128), mode='bilinear')
        other = F.interpolate(x[:,3:], size=(128, 128), mode='nearest')
        return torch.cat((rgb,other), 1)

    if DiffAugment==False:
        rgb = F.interpolate(x[:,:3], size=(128, 128), mode='bilinear')
        other = F.interpolate(x[:,3:], size=(128, 128), mode='nearest')
        return torch.cat((rgb,other), 1)

    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, isTrue=isTrue)
        rgb = F.interpolate(x[:,:3], size=(128, 128), mode='bilinear')
        other = F.interpolate(x[:,3:], size=(128, 128), mode='nearest')
        x = torch.cat((rgb,other), 1)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    if x.shape[1] > 3:
        x[:,:3] = x[:,:3] + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    else:
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    if x.shape[1] > 3:
        x_mean = x[:,:3].mean(dim=1, keepdim=True)
        x[:,:3] = (x[:,:3] - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                       dtype=x.dtype, device=x.device) * 2) + x_mean
    else:
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                       dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    if x.shape[1] > 3:
        x_mean = x[:,:3].mean(dim=[1, 2, 3], keepdim=True)
        x[:,:3] = (x[:,:3] - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                 dtype=x.dtype, device=x.device) + 0.5) + x_mean
    else:
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                   dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_flip(rgb, min_size=64, max_size=256, isTrue=False):

    H_idx = np.random.randint(0,100)
    if H_idx > 50:
        rgb = kornia.geometry.transform.hflip(rgb)
    H_idx = np.random.randint(0,100)
    if H_idx > 50:
        rgb = kornia.geometry.transform.vflip(rgb)
    return rgb





def rand_affine(rgb):
    H_idx = np.random.randint(0,100)
    if H_idx > 50:
        angles_rad = torch.rand(rgb.shape[0]) * K.pi
        angles_deg = kornia.rad2deg(angles_rad) * 360
        params = {'angle': angles_deg, 'center': (64,64), 'resample': 0, 'align_corners':True, 'sx': (torch.rand(rgb.shape[0]) - 0.5)*0.2,'sy': (torch.rand(rgb.shape[0]) - 0.5)*0.2}
        x_data: torch.Tensor = rgb.view(-1, *rgb.shape[-3:])
        height, width = x_data.shape[-2:]
        transform = compute_affine_transformation(rgb, params)
        resample_name: str = Resample(params['resample'].item()).name.lower()
        align_corners: bool = cast(bool, params['align_corners'].item())
        out_data: torch.Tensor = warp_affine(x_data, transform[:, :2, :],
                                         (height, width), resample_name,
                                         align_corners=align_corners,padding_mode='border')

    # H_idx = np.random.randint(0,100)
    # if H_idx > 50:
        # params = {'sx': (torch.rand(rgb.shape[0]) - 0.5),'sy': (torch.rand(rgb.shape[0]) - 0.5), 'center': (64,64), 'resample': 0, 'align_corners':True}
        # x_data: torch.Tensor = rgb.view(-1, *rgb.shape[-3:])
        # height, width = x_data.shape[-2:]
        # transform = compute_affine_transformation(rgb, params)
        # resample_name: str = Resample(params['resample'].item()).name.lower()
        # align_corners: bool = cast(bool, params['align_corners'].item())
        # out_data: torch.Tensor = warp_affine(x_data, transform[:, :2, :],
                                         # (height, width), resample_name,
                                         # align_corners=align_corners,padding_mode='border')

    return out_data
       # aug = RandomAffine((-15., 20.), return_transform=True, resample="NEAREST")


def rand_translation(rgb, min_size=64, max_size=256, isTrue=False):
    H, W = rgb.shape[2:]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    # H_size_list = [16, 32, 64, 128, 128]
    # H_size = H_size_list[H_idx]
    if isTrue ==True:
        # H_idx = np.random.randint(128,256)
        # H_size = np.random.randint(128,256)
        H_idx = 128
        H_size = 128
    else:
        H_idx = 128#np.random.randint(32, 64)
        H_size = 128#np.random.randint(32, 64)
    x = 8
    if ((H_size | (x - 1)) + 1) > min_HW:
        H_size = H_size - x

    W_size = H_size
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    mask = rgb[0].sum(0).detach().cpu().reshape(H, W)
    coord = np.argwhere(mask != 0).T
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    rgb = rgb[:, :, begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    return rgb

# def rand_translation(x, ratio=0.125):
    # shift_x, shift_y = int(x.size(2) * ratio +
                           # 0.5), int(x.size(3) * ratio + 0.5)
    # translation_x = torch.randint(-shift_x, shift_x + 1,
                                  # size=[x.size(0), 1, 1], device=x.device)
    # translation_y = torch.randint(-shift_y, shift_y + 1,
                                  # size=[x.size(0), 1, 1], device=x.device)
    # grid_batch, grid_x, grid_y = torch.meshgrid(
        # torch.arange(x.size(0), dtype=torch.long, device=x.device),
        # torch.arange(x.size(2), dtype=torch.long, device=x.device),
        # torch.arange(x.size(3), dtype=torch.long, device=x.device),
    # )
    # grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    # grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    # x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    # x = x_pad.permute(0, 2, 3, 1).contiguous()[
        # grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    # return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(
        2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(
        3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x -
                         cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y -
                         cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    # 'color': [rand_brightness, rand_saturation, rand_contrast],
    # 'translation': [rand_affine, rand_translation, rand_flip],
    'translation': [rand_translation, rand_flip],
    # 'cutout': [rand_cutout],
}
