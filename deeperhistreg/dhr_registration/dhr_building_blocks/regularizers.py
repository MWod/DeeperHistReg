### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Callable, Union

### External Imports ###
import torch as tc

### Internal Imports ###


########################

def diffusion(
    displacement_field: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Diffusion regularization (PyTorch).
    Parameters
    ----------
    displacement_field : tc.Tensor
        The input displacment field (2-D or 3-D) (B x size x ndim)
    params : dict
        Additional parameters
    Returns
    ----------
    diffusion_reg : float
        The value denoting the decrease of displacement field smoothness
    """
    ndim = len(displacement_field.size()) - 2
    if ndim == 2:
        dx = (displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])**2
        dy = (displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy)) / 2
    elif ndim == 3:
        dx = (displacement_field[:, 1:, :, :, :] - displacement_field[:, :-1, :, :, :])**2
        dy = (displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :])**2
        dz = (displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy) + tc.mean(dz)) / 3
    else:
        raise ValueError("Unsupported number of dimensions.")
    return diffusion_reg

def diffusion_relative(
    displacement_field: tc.Tensor,
    device: Union[str, tc.device, None]="cpu",
    **params : dict) -> tc.Tensor:
    """
    Relative diffusion regularization (with respect to the input size) (PyTorch).
    Parameters
    ----------
    displacement_field : tc.Tensor
        The input displacment field (2-D or 3-D) (B x size x ndim)
    params : dict
        Additional parameters
    Returns
    ----------
    diffusion_reg : float
        The value denoting the decrease of displacement field smoothness
    """
    ndim = len(displacement_field.size()) - 2
    if ndim == 2:
        dx = ((displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])*displacement_field.shape[2])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy)) / 2
    elif ndim == 3:
        dx = ((displacement_field[:, 1:, :, :, :] - displacement_field[:, :-1, :, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :])*displacement_field.shape[2])**2
        dz = ((displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :])*displacement_field.shape[3])**2
        diffusion_reg = (tc.mean(dx) + tc.mean(dy) + tc.mean(dz)) / 3
    else:
        raise ValueError("Unsupported number of dimensions.")
    return diffusion_reg

def get_function(function_name : str) -> Callable:
    return getattr(current_file, function_name)