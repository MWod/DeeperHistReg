### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt

### Internal Imports ###
from dhr_paths import model_paths as p
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_networks import superglue as sg
from dhr_registration.dhr_building_blocks import cost_functions as cf

########################

def superpoint_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    echo = params['echo']
    resolution = params['registration_size']
    try:
        return_num_matches = params['return_num_matches']
    except KeyError:
        return_num_matches = False

    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)   
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg  = u.tensor_to_image(resampled_target)[:, :, 0]

    transform, num_matches, target_keypoints = perform_registration(src, trg, params)
    final_transform = w.affine2theta(transform.astype(np.float64), (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
    if echo:
        print(f"Calculacted transform: {final_transform}")
        print(f"Number of matches: {num_matches}")
    if return_num_matches:
        return final_transform, num_matches
    else:
        return final_transform

def perform_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    default_params = {'superpoint_weights_path': p.superpoint_model_path, 'superglue_weights_path' : p.superglue_model_path,
     'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 3000, 'sinkhorn_iterations': 30, 'match_threshold': 0.3, 'show': False, 'echo': True,
     'transform_type': "affine", 'device': "cuda:0"}
    params = {**default_params, **params}
    superpoint_weights_path = params['superpoint_weights_path']
    superglue_weights_path = params['superglue_weights_path']
    nms_radius = params['nms_radius']
    keypoint_threshold = params['keypoint_threshold']
    max_keypoints = params['max_keypoints']
    superglue = 'outdoor' # artifact
    sinkhorn_iterations = params['sinkhorn_iterations']
    match_threshold = params['match_threshold']
    show = params['show']
    echo = params['echo']
    device = params['device']
    transform_type = params['transform_type']

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }       
    }
    model = sg.Matching(config).eval().to(device)
    model.superpoint.load_state_dict(tc.load(superpoint_weights_path))
    model.superglue.load_state_dict(tc.load(superglue_weights_path))

    source = tc.from_numpy(source).unsqueeze(0).unsqueeze(0).to(device)
    target = tc.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device)

    pred = model({'image0': source, 'image1': target})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    if echo:
        print(f"Number of source keypoints: {len(kpts0)}")
        print(f"Number of target keypoints: {len(kpts1)}")
    matches = pred['matches0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(source.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(kpts0[:, 0], kpts0[:, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(target.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(kpts1[:, 0], kpts1[:, 1], "r*")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(source.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(mkpts0[:, 0], mkpts0[:, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(target.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(mkpts1[:, 0], mkpts1[:, 1], "r*")

        plt.figure()
        for i in range(len(mkpts0)):
            plt.plot([mkpts0[i, 0], mkpts1[i, 0]], [mkpts0[i, 1], mkpts1[i, 1]], "*-")
        plt.show()

    h_pts0 = u.points_to_homogeneous_representation(mkpts0)
    h_pts1 = u.points_to_homogeneous_representation(mkpts1)
    if echo:
        print(f"Number of matches: {len(h_pts0)}")

    if transform_type == "affine":
        transform = u.calculate_affine_transform(h_pts1, h_pts0)
    elif transform_type == "rigid":
        transform = u.calculate_rigid_transform(mkpts1, mkpts0)
    else:
        raise ValueError("Unsupported transform type (rigid or affine only).")
    return transform, len(h_pts0), mkpts1

def ransac(
    source_points : np.ndarray,
    target_points : np.ndarray,
    num_iters : int=30,
    threshold : float=10.0,
    num_points : int=3,
    transform_type : str='affine',
    echo : bool=True) -> np.ndarray:
    """
    TODO
    """
    indices = np.arange(len(source_points))
    if transform_type == "affine":
        best_transform = u.calculate_affine_transform(u.points_to_homogeneous_representation(target_points), u.points_to_homogeneous_representation(source_points))
    elif transform_type == "rigid":
        best_transform = u.calculate_rigid_transform(target_points, source_points)
    else:
        raise ValueError("Unsupported transform type (rigid or affine only).")
    best_ratio = 0.0

    for i in range(num_iters):
        current_indices = np.random.choice(indices, num_points, replace=False)
        current_sp = source_points[current_indices, :]
        current_tp = target_points[current_indices, :]
        if transform_type == "affine":
            transform = u.calculate_affine_transform(current_tp, current_sp)
        elif transform_type == "rigid":
            transform = u.calculate_rigid_transform(current_tp, current_sp)
        else:
            raise ValueError("Unsupported transform type (rigid or affine only).")
        transformed_target_points = (transform @ u.points_to_homogeneous_representation(target_points).swapaxes(1, 0)).swapaxes(0, 1)
        error = ((u.points_to_homogeneous_representation(source_points) - transformed_target_points)**2).mean(axis=1)
        inliers = (error < threshold).sum()
        ratio = inliers / len(source_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_transform = transform
            if echo:
                print(f"Current best ratio: {best_ratio}")
    return best_transform