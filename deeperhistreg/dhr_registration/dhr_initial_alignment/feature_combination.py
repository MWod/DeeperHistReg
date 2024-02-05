### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import torch as tc
import cv2

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_building_blocks import cost_functions as cf

import superpoint_ransac as spr
import sift_ransac as sr
import superpoint_superglue as sg

########################

def feature_based_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    transforms = []
    registration_sizes = params['registration_sizes']
    for registration_size in registration_sizes:
        ex_params = {**params, **{'registration_size': registration_size}}
        current_transform = sr.sift_ransac(source, target, ex_params)
        transforms.append(current_transform)
        current_transform = sg.superpoint_superglue(source, target, ex_params)
        transforms.append(current_transform)
        current_transform = spr.superpoint_ransac(source, target, ex_params)
        transforms.append(current_transform)

    resolution = params['registration_size']
    echo = params['echo']
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution) 

    # TODO - add evaluation mode based on the number of matched pairs instead of sparse keypoint (as option)

    sift = cv2.SIFT_create(256)
    keypoints, _ = sift.detectAndCompute((resampled_target[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), None)
    if echo:
        print(f"Number of evaluation keypoints: {len(keypoints)}")

    best_transform = tc.eye(3, device=source.device)[0:2, :].unsqueeze(0)
    best_cost = cf.sparse_ncc(resampled_source, resampled_target, **{'keypoints': keypoints, 'win_size': 45}) # 45
    if echo:
        print(f"Initial cost: {best_cost}")
    for transform in transforms:
        displacement_field = w.tc_transform_to_tc_df(transform, resampled_source.size())
        warped_source = w.warp_tensor(resampled_source, displacement_field)
        current_cost = cf.sparse_ncc(warped_source, resampled_target, **{'keypoints': keypoints, 'win_size': 45})
        if echo:
            print(f"Current cost: {current_cost}")
        if current_cost < best_cost:
            best_cost = current_cost
            best_transform = transform
            if echo:
                print(f"Current best: {best_cost}") 
    if echo:
        print(f"Current best: {best_cost}")
        print(f"Final transform: {best_transform}")
    return best_transform


def rotated_feature_based_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    angle_step = params['angle_step']
    device = params['device']
    resolution = params['registration_size']
    echo = params['echo']
    num_features = params['num_features']
    sparse_size = params['sparse_size']
    keypoint_size = params['keypoint_size'] # 8
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution) 

    # TODO - add evaluation mode based on the number of matched pairs instead of sparse keypoint (as option)

    sift = cv2.SIFT_create(num_features) #256
    keypoints, target_descriptors = sift.detectAndCompute((resampled_target[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), None)
    if echo:
        print(f"Number of evaluation keypoints: {len(keypoints)}")

    best_transform = tc.eye(3, device=source.device)[0:2, :].unsqueeze(0)
    # best_cost = cf.sparse_ncc_tc(resampled_source, resampled_target, **{'keypoints': keypoints, 'win_size': sparse_size}) # 45
    _, source_descriptors = sift.compute((resampled_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
    costs = np.mean((source_descriptors - target_descriptors)**2, axis=1)
    lowest_costs = np.sort(costs)[0:keypoint_size]
    best_cost = np.mean(lowest_costs)

    angle_start, angle_stop = -180, 180
    for angle in range(angle_start, angle_stop, angle_step):
        _, _, y_size, x_size = source.shape
        x_origin = x_size // 2 
        y_origin = y_size // 2
        r_transform = w.generate_rigid_matrix(angle, x_origin, y_origin, 0, 0)
        r_transform = w.affine2theta(r_transform, (source.size(2), source.size(3))).to(device).unsqueeze(0)
        current_displacement_field = w.tc_transform_to_tc_df(r_transform, (1, 1, source.size(2), source.size(3)))
        transformed_source = w.warp_tensor(source, current_displacement_field)

        transforms = []
        registration_sizes = params['registration_sizes']
        for registration_size in registration_sizes:
            ex_params = {**params, **{'registration_size': registration_size}}
            current_transform = sr.sift_ransac(transformed_source, target, ex_params)
            transforms.append(current_transform)
            current_transform = sg.superpoint_superglue(transformed_source, target, ex_params)
            transforms.append(current_transform)
            current_transform = spr.superpoint_ransac(transformed_source, target, ex_params)
            transforms.append(current_transform)

        if echo:
            print(f"Initial cost: {best_cost}")

        for transform in transforms:
            transform = w.compose_transforms(r_transform[0], transform).unsqueeze(0).to(device)
            displacement_field = w.tc_transform_to_tc_df(transform, resampled_source.size())
            warped_source = w.warp_tensor(resampled_source, displacement_field)
            # current_cost = cf.sparse_ncc_tc(warped_source, resampled_target, **{'keypoints': keypoints, 'win_size': sparse_size})
            source_keypoints, source_descriptors = sift.compute((warped_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
            # current_cost = np.mean((source_descriptors - target_descriptors)**2)
            costs = np.mean((source_descriptors - target_descriptors)**2, axis=1)
            lowest_costs = np.sort(costs)[0:keypoint_size]
            current_cost = np.mean(lowest_costs)
            if echo:
                print(f"Current cost: {current_cost}")
            if current_cost < best_cost:
                best_cost = current_cost
                best_transform = transform
                if echo:
                    print(f"Current best: {best_cost}") 
    if echo:
        print(f"Current best: {best_cost}")
        print(f"Final transform: {best_transform}")
    return best_transform