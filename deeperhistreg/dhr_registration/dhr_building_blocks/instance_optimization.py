### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Callable, Union

### External Imports ###
import torch as tc
import torch.nn.functional as F
import torch.optim as optim

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w
import bsplines as bs

########################


def affine_registration(
    source: tc.Tensor,
    target: tc.Tensor,
    num_levels: int,
    used_levels: int,
    num_iters: list,
    learning_rate: float,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float],
    cost_function_params: dict={}, 
    device: Union[str, tc.device, None]="cpu",
    initial_transform : Union[tc.Tensor, None]=None,
    echo: bool=False,
    return_best: bool=False) -> tc.Tensor:
    """
    Performs the affine registration using the instance optimization technique (a prototype).

    Parameters
    ----------
    source : tc.Tensor
        The source tensor (1x1 x size)
    target : tc.Tensor
        The target tensor (1x1 x size)
    num_levels : int
        The number of resolution levels
    used_levels : int
        The number of actually used resolution levels (must be lower (or equal) than the num_levels)
    num_iters : int
        The nubmer of iterations per resolution
    learning_rate : float
        The learning rate for the optimizer
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    cost_function_params : dict (default: {})
        The optional cost function parameters
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    Returns
    ----------
    transformation : tc.Tensor
        The affine transformation matrix (1 x transformation_size (2x3 or 3x4))
    """
    ndim = len(source.size()) - 2
    if initial_transform is None:
        if ndim == 2:
            transformation = tc.zeros((1, 2, 3), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        elif ndim == 3:
            transformation = tc.zeros((1, 3, 4), dtype=source.dtype, device=device)
            transformation[0, 0, 0] = 1.0
            transformation[0, 1, 1] = 1.0
            transformation[0, 2, 2] = 1.0
            transformation = transformation.detach().clone()
            transformation.requires_grad = True
        else:
            raise ValueError("Unsupported number of dimensions.")
    else:
        transformation = initial_transform.detach().clone()
        transformation.requires_grad = True

    optimizer = optim.Adam([transformation], learning_rate)
    source_pyramid = u.create_pyramid(source, num_levels=num_levels)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels)
    if return_best:
        best_transformation = transformation.clone()
        best_cost = 1000.0
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                sampling_grid = F.affine_grid(transformation, size=current_source.size(), align_corners=False)
                warped_source = w.transform_tensor(current_source, sampling_grid, device=device)
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)    
                cost.backward()
                optimizer.step()
                current_cost = cost.item()
            optimizer.zero_grad()
            if echo:
                print(f"Iter: {i}, Current cost: {current_cost}")
            if return_best:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_transformation = transformation.clone()
    if return_best:
        return best_transformation
    else:
        return transformation


def nonrigid_registration(
    source: tc.Tensor,
    target: tc.Tensor,
    num_levels: int,
    used_levels: int,
    num_iters: list,
    learning_rates: list,
    alphas: list,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float],
    regularization_function: Callable[[tc.Tensor, dict], float],
    cost_function_params: dict={},
    regularization_function_params: dict={},
    penalty_function: Callable=None,
    penalty_function_params: dict={},
    initial_displacement_field: Union[tc.Tensor, None]=None,
    device: Union[str, tc.device, None]="cpu",
    echo: bool=False) -> tc.Tensor:
    """
    Performs the nonrigid registration using the instance optimization technique (a prototype).

    Parameters
    ----------
    source : tc.Tensor
        The source tensor (1x1 x size)
    target : tc.Tensor
        The target tensor (1x1 x size)
    num_levels : int
        The number of resolution levels
    used_levels : int
        The number of actually used resolution levels (must be lower (or equal) than the num_levels)
    num_iters : int
        The nubmer of iterations per resolution
    learning_rate : float
        The learning rate for the optimizer
    alpha : float
        The regularization weight
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    regularization_function : Callable[tc.Tensor,  dict] -> float
        The regularization function
    cost_function_params : dict (default: {})
        The optional cost function parameters
    regularization_function_params : dict (default: {})
        The optional regularization function parameters
    penalty_function : Callable
        The optional penalty function (must be differntiable)
    penalty_function_params : dict(default: {})
        The optional penalty function parameters
    initial_displacement_field : tc.Tensor (default None)
        The initial displacement field (e.g. resulting from the initial, affine registration)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    Returns
    ----------
    displacement_field : tc.Tensor
        The calculated displacement_field (to be applied using warp_tensor from utils_tc)
    """
    source_pyramid = u.create_pyramid(source, num_levels=num_levels)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels)
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        if j == 0:
            if initial_displacement_field is None:
                displacement_field = u.create_identity_displacement_field(current_source).detach().clone()
                displacement_field.requires_grad = True
            else:
                displacement_field = u.resample_displacement_field_to_size(initial_displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
                displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rates[j])
        else:
            displacement_field = u.resample_displacement_field_to_size(displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
            displacement_field.requires_grad = True
            optimizer = optim.Adam([displacement_field], learning_rates[j])

        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                warped_source = w.warp_tensor(current_source, displacement_field, device=device)
                if i == 0:
                    if echo:
                        print(f"Initial cost: {cost_function(current_source, current_target, device=device, **cost_function_params)}")
                        print(f"First warp cost: {cost_function(warped_source, current_target, device=device, **cost_function_params)}")
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                reg = regularization_function(displacement_field, device=device, **regularization_function_params)
                loss = cost + alphas[j]*reg
                if penalty_function is not None:
                    loss = loss + penalty_function(penalty_function_params) 
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            if echo:
                print("Iter: ", i, "Current cost: ", cost.item(), "Current reg: ", reg.item(), "Current loss: ", loss.item())
    if used_levels != num_levels:
        displacement_field = u.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)))
    return displacement_field


def bsplines_registration(
    source: tc.Tensor,
    target: tc.Tensor,
    num_levels: int,
    used_levels: int,
    num_iters: list,
    learning_rates: list,
    alphas: list,
    cp_spacing: tuple,
    splines_type: str,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float],
    regularization_function: Callable[[tc.Tensor, dict], float],
    cost_function_params: dict={},
    regularization_function_params: dict={},
    penalty_function: Callable=None,
    penalty_function_params: dict={},
    device: Union[str, tc.device, None]="cpu",
    echo: bool=False) -> tc.Tensor:
    """
    TODO - documentation
    """
    source_pyramid = u.create_pyramid(source, num_levels=num_levels)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels)
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        if j == 0:
            control_points = bs.create_control_points(current_source, cp_spacing, splines_type=splines_type, normalize=True)
            cp_copy = control_points.clone().detach()
            control_points.requires_grad = True
            optimizer = optim.Adam([control_points], learning_rates[j])
        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                displacement_field = bs.b_splines(current_source, control_points, cp_spacing, splines_type=splines_type, normalize=True)
                warped_source = w.warp_tensor(current_source, displacement_field, device=device)
                if i == 0:
                    if echo:
                        print(f"Initial cost: {cost_function(current_source, current_target, device=device, **cost_function_params)}")
                        print(f"First warp cost: {cost_function(warped_source, current_target, device=device, **cost_function_params)}")
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                reg = regularization_function(control_points, device=device, **regularization_function_params)
                frame = tc.mean((control_points[:, 0, :, :]  - cp_copy[:, 0, :, :])**2 + (control_points[:, -1, :, :]  - cp_copy[:, -1, :, :])**2) + tc.mean((control_points[:, :, 0, :]  - cp_copy[:, :, 0, :])**2 + (control_points[:, :, -1, :]  - cp_copy[:, :, -1, :])**2)
                loss = cost + alphas[j]*reg + 1000*frame
                if penalty_function is not None:
                    loss = loss + penalty_function(penalty_function_params) 
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            if echo:
                print("Iter: ", i, "Current cost: ", cost.item(), "Current reg: ", reg.item(), "Current loss: ", loss.item())
    return control_points

def nonrigid_registration_lbfgs(
    source: tc.Tensor,
    target: tc.Tensor,
    num_levels: int,
    used_levels: int,
    num_iters: list,
    alphas: list,
    cost_function: Callable[[tc.Tensor, tc.Tensor, dict], float],
    regularization_function: Callable[[tc.Tensor, dict], float],
    cost_function_params: dict={},
    regularization_function_params: dict={},
    penalty_function: Callable=None,
    penalty_function_params: dict={},
    initial_displacement_field: Union[tc.Tensor, None]=None,
    device: Union[str, tc.device, None]="cpu",
    echo: bool=False) -> tc.Tensor:
    """
    Performs the nonrigid registration using the instance optimization technique (a prototype).

    Parameters
    ----------
    source : tc.Tensor
        The source tensor (1x1 x size)
    target : tc.Tensor
        The target tensor (1x1 x size)
    num_levels : int
        The number of resolution levels
    used_levels : int
        The number of actually used resolution levels (must be lower (or equal) than the num_levels)
    num_iters : int
        The nubmer of iterations per resolution
    alpha : float
        The regularization weight
    cost_function : Callable[tc.Tensor, tc.Tensor, dict] -> float
        The cost function being optimized
    regularization_function : Callable[tc.Tensor,  dict] -> float
        The regularization function
    cost_function_params : dict (default: {})
        The optional cost function parameters
    regularization_function_params : dict (default: {})
        The optional regularization function parameters
    penalty_function : Callable
        The optional penalty function (must be differntiable)
    penalty_function_params : dict(default: {})
        The optional penalty function parameters
    initial_displacement_field : tc.Tensor (default None)
        The initial displacement field (e.g. resulting from the initial, affine registration)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")

    TODO - add params for LFBGS

    Returns
    ----------
    displacement_field : tc.Tensor
        The calculated displacement_field (to be applied using warp_tensor from utils_tc)
    """
    source_pyramid = u.create_pyramid(source, num_levels=num_levels)
    target_pyramid = u.create_pyramid(target, num_levels=num_levels)
    for j in range(used_levels):
        current_source = source_pyramid[j]
        current_target = target_pyramid[j]
        if j == 0:
            if initial_displacement_field is None:
                displacement_field = u.create_identity_displacement_field(current_source).detach().clone()
                displacement_field.requires_grad = True
            else:
                displacement_field = u.resample_displacement_field_to_size(initial_displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
                displacement_field.requires_grad = True
            optimizer = optim.LBFGS([displacement_field], tolerance_grad=1e-13, tolerance_change=1e-15, history_size=50, max_iter=4, line_search_fn=None)
        else:
            displacement_field = u.resample_displacement_field_to_size(displacement_field, (current_source.size(2), current_source.size(3))).detach().clone()
            displacement_field.requires_grad = True
            optimizer = optim.LBFGS([displacement_field], tolerance_grad=1e-13, tolerance_change=1e-15, history_size=50, max_iter=4, line_search_fn=None)

        for i in range(num_iters[j]):
            with tc.set_grad_enabled(True):
                warped_source = w.warp_tensor(current_source, displacement_field, device=device)
                if i == 0:
                    if echo:
                        print(f"Initial cost: {cost_function(current_source, current_target, device=device, **cost_function_params)}")
                        print(f"First warp cost: {cost_function(warped_source, current_target, device=device, **cost_function_params)}")
                cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                reg = regularization_function(displacement_field, device=device, **regularization_function_params)
                loss = cost + alphas[j]*reg
                if penalty_function is not None:
                    loss = loss + penalty_function(penalty_function_params) 
                loss.backward()
                def step():
                    warped_source = w.warp_tensor(current_source, displacement_field, device=device)
                    cost = cost_function(warped_source, current_target, device=device, **cost_function_params)   
                    reg = regularization_function(displacement_field, device=device, **regularization_function_params)
                    loss = cost + alphas[j]*reg
                    return loss
                optimizer.step(step)
            optimizer.zero_grad()
            if echo:
                print("Iter: ", i, "Current cost: ", cost.item(), "Current reg: ", reg.item(), "Current loss: ", loss.item())
    if used_levels != num_levels:
        displacement_field = u.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)))
    return displacement_field