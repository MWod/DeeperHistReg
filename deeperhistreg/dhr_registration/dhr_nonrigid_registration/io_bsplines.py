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

"""
Note: The method outputs the control control points mapping without composing with the initial displacement field.
Therefore, if you use initial_displacement_field parameter other than None, you have to convert the control points
into the displacement field and compose it with the initial displacement field before use.
"""

def instance_optimization_bsplines_registration(
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
    cp_spacing = params['cp_spacing']
    splines_type = params['splines_type']

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

    ### BSplines Registration ###
    if initial_displacement_field is None:
        initial_source = resampled_source
        control_points = io.bsplines_registration(initial_source, resampled_target, num_levels, used_levels, iterations, learning_rates, alphas, cp_spacing, splines_type,
            cost_function, regularization_function, cost_function_params, regularization_function_params, device=device, echo=echo)
    else:
        initial_df = u.resample_displacement_field_to_size(initial_displacement_field, (resampled_source.size(2), resampled_source.size(3)))
        with tc.set_grad_enabled(False):
            initial_source = w.warp_tensor(resampled_source, initial_df)
        control_points = io.bsplines_registration(initial_source, resampled_target, num_levels, used_levels, iterations, learning_rates, alphas, cp_spacing, splines_type,
            cost_function, regularization_function, cost_function_params, regularization_function_params, device=device, echo=echo) 
    return control_points