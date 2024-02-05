### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union

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

def tre(source_landmarks : Union[tc.Tensor, np.ndarray], target_landmarks : Union[tc.Tensor, np.ndarray]):
    # TODO - documentation
    if isinstance(source_landmarks, tc.Tensor) and isinstance(target_landmarks, tc.Tensor):
        return tc.sqrt(((source_landmarks - target_landmarks)**2).sum(axis=1))
    elif isinstance(source_landmarks, np.ndarray) and isinstance(target_landmarks, np.ndarray):
        return np.sqrt(((source_landmarks - target_landmarks)**2).sum(axis=1))
    else:
        raise ValueError("Unsupported type.")

def robustness(source_landmarks, target_landmarks, warped_target_landmarks):
    # TODO - documentation
    return (tre(source_landmarks, warped_target_landmarks) < tre(source_landmarks, target_landmarks)).mean()

def jacobian_determinant(displacement_field : np.ndarray):
    # TODO - documentation
    # TODO - create version for tc.Tensor
    _, y_size, x_size = np.shape(displacement_field)
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    x_field, y_field = x_grid + displacement_field[0], y_grid + displacement_field[1]
    u_x_grad, u_y_grad = np.gradient(x_field), np.gradient(y_field)
    u_xy, u_xx = u_x_grad[0], u_x_grad[1]
    u_yy, u_yx = u_y_grad[0], u_y_grad[1]
    jac = np.array([[u_xx, u_xy], [u_yx, u_yy]]).swapaxes(1, 2).swapaxes(0, 3).swapaxes(0, 1)
    jac_det = np.linalg.det(jac)
    return jac_det

def folding_ratio(jacobian_determinant : np.ndarray):
    # TODO - documentation
    # TODO - create version for tc.Tensor
    return np.mean(jacobian_determinant < 0) * 100

def jacobian_det_stddev(jacobian_determinant : np.ndarray):
    # TODO - documentation
    # TODO - create version for tc.Tensor
    jac_det = (jacobian_determinant + 15).clip(0.0000000001, 1000000000)
    log_jac_det = np.std(np.log(jac_det))
    return log_jac_det
