### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import torch as tc

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_building_blocks import cost_functions as cf
from dhr_building_blocks import instance_optimization as io

########################


def exhaustive_rigid_search(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO - documentation 
    Assumes images with [0-1] input range where 0 is the background.
    """
    device = params['device']
    echo = params['echo']
    angle_step = params['angle_step']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    resolution = params['registration_size']
    affine_iters = params['affine_iters']

    if type(cost_function) == str:
        cost_function = cf.get_function(cost_function)

    ### Initial resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")
    initial_cost_function = cost_function(resampled_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Initial objective function: {initial_cost_function.item()}")

    ### Center of mass calculation and warping ###
    com_x_source, com_y_source = u.center_of_mass(resampled_source)
    com_x_target, com_y_target = u.center_of_mass(resampled_target)
    if echo:
        print(f"Source center of mass: {(com_y_source, com_x_source)}")
        print(f"Target center of mass: {(com_y_target, com_x_target)}")

    centroid_transform = w.generate_rigid_matrix(0, 0, 0, com_x_source - com_x_target, com_y_source - com_y_target)
    centroid_transform = w.affine2theta(centroid_transform, (resampled_source.size(2), resampled_source.size(3))).unsqueeze(0).to(device)
    centroid_df = w.tc_transform_to_tc_df(centroid_transform, (1, 1, resampled_source.size(2), resampled_source.size(3)))
    translated_source = w.warp_tensor(resampled_source, centroid_df)
    centroid_cost_function = cost_function(translated_source, resampled_target, device=device, **cost_function_params)
    if echo:
        print(f"Centroid-aligned objective function: {centroid_cost_function.item()}")

    ### Exhaustive rotation alignment ###
    best_cost_function = centroid_cost_function
    found = False
    angle_start, angle_stop = 0, 360
    with tc.set_grad_enabled(False):
        for i in range(angle_start, angle_stop, angle_step):
            transform = w.generate_rigid_matrix(i, com_x_target, com_y_target, 0, 0)
            transform = w.affine2theta(transform, (resampled_source.size(2), resampled_source.size(3))).to(device)
            transform = w.compose_transforms(centroid_transform[0], transform).unsqueeze(0).to(device)
            transform = io.affine_registration(resampled_source, resampled_target, 2, 2, [affine_iters, affine_iters], 0.001, cost_function, cost_function_params, device=device, initial_transform=transform, echo=False, return_best=True)
            current_displacement_field = w.tc_transform_to_tc_df(transform, (1, 1, resampled_source.size(2), resampled_source.size(3)))
            transformed_source = w.warp_tensor(resampled_source, current_displacement_field)
            
            current_cost_function = cost_function(transformed_source, resampled_target, device=device, **cost_function_params)
            if echo:
                print(f"Angle: {i}, Cost function: {current_cost_function}")
            if current_cost_function < best_cost_function:
                found = True
                best_cost_function = current_cost_function
                best_transform = transform
    if echo:
        print(f"Best cost function: {best_cost_function}")

    ### Final processing of the output transform ###
    if found:
        to_return = best_transform
    else:
        if centroid_cost_function < initial_cost_function:
            to_return = centroid_transform
        else:
            to_return = tc.eye(3, device=device)
    if echo:
        print(f"Final transform: {to_return}")
    return to_return