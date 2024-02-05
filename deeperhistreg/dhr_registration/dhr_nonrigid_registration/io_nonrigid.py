### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union

### External Imports ###
import torch as tc

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_building_blocks import regularizers as rg
from dhr_building_blocks import cost_functions as cf
from dhr_building_blocks import instance_optimization as io

########################

def instance_optimization_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    device = params['device']
    echo = params['echo']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    regularization_function = params['regularization_function']
    regularization_function_params = params['regularization_function_params']
    resolution = params['registration_size']

    num_levels = params['num_levels']
    used_levels = params['used_levels']
    iterations = params['iterations']
    learning_rates = params['learning_rates']
    alphas = params['alphas']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)
    if type(regularization_function) == str:
        regularization_function = rg.get_function(regularization_function)

    ### Initial Resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### Nonrigid Registration ###
    if initial_displacement_field is None:
        initial_df = None
        displacement_field = io.nonrigid_registration(resampled_source, resampled_target, num_levels, used_levels, iterations, learning_rates, alphas,
            cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=initial_df, device=device, echo=echo)
    else:
        initial_df = u.resample_displacement_field_to_size(initial_displacement_field, (resampled_source.size(2), resampled_source.size(3)))
        with tc.set_grad_enabled(False):
            warped_source = w.warp_tensor(resampled_source, initial_df, mode='bicubic')
        displacement_field = io.nonrigid_registration(warped_source, resampled_target, num_levels, used_levels, iterations, learning_rates, alphas,
            cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=None, device=device, echo=echo)
        displacement_field = w.compose_displacement_fields(initial_df, displacement_field)

    if echo:
        print(f"Registered displacement field size: {displacement_field.size()}")
    displacement_field = u.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)), mode='bicubic')
    if echo:
        print(f"Output displacement field size: {displacement_field.size()}")
    return displacement_field


def instance_optimization_nonrigid_registration_lbfgs(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    device = params['device']
    echo = params['echo']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    regularization_function = params['regularization_function']
    regularization_function_params = params['regularization_function_params']
    resolution = params['registration_size']

    num_levels = params['num_levels']
    used_levels = params['used_levels']
    iterations = params['iterations']
    alphas = params['alphas']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)
    if type(regularization_function) == str:
        regularization_function = rg.get_function(regularization_function)

    ### Initial resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### Nonrigid Registration ###
    if initial_displacement_field is None:
        initial_df = None
        displacement_field = io.nonrigid_registration_lbfgs(resampled_source, resampled_target, num_levels, used_levels, iterations, alphas,
            cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=initial_df, device=device, echo=echo)
    else:
        initial_df = u.resample_displacement_field_to_size(initial_displacement_field, (resampled_source.size(2), resampled_source.size(3)))
        with tc.set_grad_enabled(False):
            warped_source = w.warp_tensor(resampled_source, initial_df, mode='bicubic')
        displacement_field = io.nonrigid_registration_lbfgs(warped_source, resampled_target, num_levels, used_levels, iterations, alphas,
            cost_function, regularization_function, cost_function_params, regularization_function_params, initial_displacement_field=None, device=device, echo=echo)
        displacement_field = w.compose_displacement_fields(initial_df, displacement_field)

    if echo:
        print(f"Registered displacement field size: {displacement_field.size()}")
    displacement_field = u.resample_displacement_field_to_size(displacement_field, (source.size(2), source.size(3)), mode='bicubic')
    if echo:
        print(f"Output displacement field size: {displacement_field.size()}")
    return displacement_field
