#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import wandb

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def normalize_to_distribution(tensor):
    flat_tensor = tensor.view(tensor.shape[0], -1)  # (batch_size, num_pixels)
    normalized_tensor = F.softmax(flat_tensor, dim=1)
    return normalized_tensor.view(tensor.shape) 

def kl_divergence_loss(output, gt):
    output_prob = normalize_to_distribution(output)
    gt_prob = normalize_to_distribution(gt)
    log_output_prob = torch.log(output_prob) 
    kl_loss = F.kl_div(log_output_prob, gt_prob, reduction='batchmean')
    return kl_loss


def down_scale_splatter(splatter, k ,v, resolution):
    gt_reshped = splatter[k][0].view(128, 128, splatter[k][0].shape[-1]).permute(2, 0, 1).unsqueeze(0)
    gt_reshped = F.interpolate(gt_reshped, size=(resolution, resolution), mode='bilinear')
    gt_reshped = gt_reshped.squeeze(0).permute(1, 2, 0)
    if v.dim() == 3 and v.size(1) == 1:
        gt_reshped = gt_reshped.view(v.size(0), 1, v.size(-1))
    else:
        gt_reshped = gt_reshped.view(v.size(0), v.size(-1))

    return gt_reshped

def filter_by_opacity(image_dict, std_factor=1.0):
    filtered_dict = {}
    opacity = image_dict['opacity'].flatten()
    opacity_mean = opacity.mean()
    opacity_std = opacity.std()
    opacity_threshold = opacity_mean + std_factor * opacity_std

    opacity_threshold = max(0.1, opacity_threshold)

    mask = opacity > opacity_threshold
    mask = mask.unsqueeze(-1) 

    for k, v in image_dict.items():
        if k == 'xyz':
            filtered_v = (v.view(-1, v.shape[-1]) * mask).view(v.shape)
            filtered_dict[k] = filtered_v
        else:
            filtered_dict[k] = v.clone()  # No change needed for other splatters

    return filtered_dict




def splatter_image_loss(network_output, gt, training_resolution):
    loss_total = 0
    #gt_filtered = filter_by_opacity(gt)
    #network_output = filter_by_opacity(network_output)
    gt_filtered = gt
    for k, v in network_output.items():
        if k != 'features_rest' and k != 'rotation':
            if training_resolution == 64:
                gt_reshped = down_scale_splatter(gt_filtered, k, v, 64)
            if training_resolution == 128:
                # dont need to downscale for 128 resolution
                gt_reshped = gt_filtered[k][0]
            loss = F.smooth_l1_loss(v, gt_reshped)
            #loss = l2_loss(v, gt_reshped)
            kl_loss_value = kl_divergence_loss(v, gt_reshped)
            v_magnitude = torch.mean(torch.abs(v)) + 1e-8
            loss_total += (0.1 * loss + 0.9* kl_loss_value)/v_magnitude

    return loss_total / (len(network_output) - 2)   # Average loss over all splatters (best run no minus 2)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

