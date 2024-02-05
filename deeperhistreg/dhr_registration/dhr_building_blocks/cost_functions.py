### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import math
import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import torch.nn.functional as F


### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w

########################

def ncc_local(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]=None,
    **params : dict) -> tc.Tensor:
    """
    Local normalized cross-correlation (as cost function) using PyTorch tensors.

    Implementation inspired by VoxelMorph (with some modifications).

    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    
    """
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 3
    try:
        mask = params['mask']
    except:
        mask = None
    window = (win_size, ) * ndim
    if device is None:
        sum_filt = tc.ones([1, 1, *window]).type_as(sources)
    else:
        sum_filt = tc.ones([1, 1, *window], device=device)

    if mask is not None:
        targets = targets * mask

    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)

def sparse_ncc(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]=None,
    **params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    keypoints = params['keypoints']
    win_size = params['win_size']
    scores = tc.zeros(len(keypoints), device=sources.device)
    _, _, y_size, x_size = sources.shape
    for i in range(len(keypoints)):
        try:
            keypoint = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])
        except:
            keypoint = int(keypoints[i, 0]), int(keypoints[i, 1])
        b_y, e_y = max(min(keypoint[1] - int(win_size // 2), y_size), 0), max(min(keypoint[1] + int(win_size // 2) + 1, y_size), 0)
        b_x, e_x = max(min(keypoint[0] - int(win_size // 2), x_size), 0), max(min(keypoint[0] + int(win_size // 2) + 1, x_size), 0)
        cs = sources[:, :, b_y:e_y, b_x:e_x]
        ts = targets[:, :, b_y:e_y, b_x:e_x]
        scores[i] = ncc_global(cs, ts)
    scores = scores[scores != 1]
    return tc.mean(scores)

def sparse_mind(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]=None,
    **params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    keypoints = params['keypoints']
    win_size = params['win_size']
    scores = tc.zeros(len(keypoints), device=sources.device)
    _, _, y_size, x_size = sources.shape
    for i in range(len(keypoints)):
        keypoint = int(keypoints[i].pt[0]), int(keypoints[i].pt[1])
        b_y, e_y = max(min(keypoint[1] - int(win_size // 2), y_size), 0), max(min(keypoint[1] + int(win_size // 2) + 1, y_size), 0)
        b_x, e_x = max(min(keypoint[0] - int(win_size // 2), x_size), 0), max(min(keypoint[0] + int(win_size // 2) + 1, x_size), 0)
        cs = sources[:, :, b_y:e_y, b_x:e_x]
        ts = targets[:, :, b_y:e_y, b_x:e_x]
        scores[i] = mind_loss(cs, ts)
    scores = scores[scores != 1]
    return tc.mean(scores)

def mse(
    sources : tc.Tensor,
    targets : tc.Tensor,
    **params : dict) -> tc.Tensor:
    return tc.mean((sources - targets)**2)

def ncc_global(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Global normalized cross-correlation (as cost function) using PyTorch tensors.

    Parameters
    ----------
    sources : tc.Tensor(Bx1xMxN)
        The source tensor
    targest : tc.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    size = sources.size()
    prod_size = tc.prod(tc.Tensor(list(size[1:])))
    sources_mean = tc.mean(sources, dim=list(range(1, len(size)))).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_mean = tc.mean(targets, dim=list(range(1, len(size)))).view((targets.size(0),) + (len(size)-1)*(1,))
    sources_std = tc.std(sources, dim=list(range(1, len(size))), unbiased=False).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_std = tc.std(targets, dim=list(range(1, len(size))), unbiased=False).view((targets.size(0),) + (len(size)-1)*(1,))
    ncc = (1 / prod_size) * tc.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=list(range(1, len(size))))
    ncc = tc.mean(ncc)
    if ncc != ncc:
        ncc = tc.autograd.Variable(tc.Tensor([-1]), requires_grad=True).to(device)
    return -ncc

def mind_loss(
    sources: tc.Tensor,
    targets: tc.Tensor,
    device: Union[str, tc.device, None]="cuda:0",
    **params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    sources = sources.view(sources.size(0), sources.size(1), sources.size(2), sources.size(3), 1)
    targets = targets.view(targets.size(0), targets.size(1), targets.size(2), targets.size(3), 1)
    try:
        dilation = params['dilation']
        radius = params['radius']
        return tc.mean((MINDSSC(sources, device=device, dilation=dilation, radius=radius) - MINDSSC(targets, device=device, dilation=dilation, radius=radius))**2)
    except:
        return tc.mean((MINDSSC(sources, device=device) - MINDSSC(targets, device=device))**2)

def pdist_squared(x : tc.Tensor) -> tc.Tensor:
    """
    TODO - documentation
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * tc.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = tc.clamp(dist.float(), 0.0, np.inf)
    return dist

def MINDSSC(
    img : tc.Tensor,
    radius : int=2,
    dilation : int=2,
    device: Union[str, tc.device, None]="cpu") -> tc.Tensor:
    """
    TODO - documentation
    """
    # Code from: https://github.com/voxelmorph/voxelmorph/pull/145 (modified to 2-D)
    kernel_size = radius * 2 + 1
    six_neighbourhood = tc.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    x, y = tc.meshgrid(tc.arange(6), tc.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = tc.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[tc.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = tc.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[tc.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = tc.nn.ReplicationPad3d(dilation)
    rpad2 = tc.nn.ReplicationPad3d(radius)
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - tc.min(ssd, 1, keepdim=True)[0]
    mind_var = tc.mean(mind, 1, keepdim=True)
    mind_var = tc.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = tc.exp(-mind)
    mind = mind[:, tc.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind


#####################################################

def get_function(function_name : str) -> Callable:
    return getattr(current_file, function_name)