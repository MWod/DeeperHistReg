### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from typing import Tuple

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import cv2

### Internal Imports ###
from dhr_paths import model_paths as p
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_building_blocks import cost_functions as cf
from dhr_networks import superglue as sg

########################


def sift_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    echo = params['echo']
    resolution = params['registration_size']
    show = params['show']

    ### Initial resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)   
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg = u.tensor_to_image(resampled_target)[:, :, 0]
    src = (src * 255).astype(np.uint8)
    trg = (trg * 255).astype(np.uint8)

    ### Descriptor calculation ###
    source_keypoints, source_descriptors, target_keypoints, target_descriptors = descriptor_calculation(src, trg)
    if echo:
        print(f"Number of source keypoints: {len(source_keypoints)}")
        print(f"Number of target keypoints: {len(target_keypoints)}")

    ### Descriptor matching below ###
    default_params = {'superglue_weights_path' : p.superglue_model_path,
     'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 3000, 'sinkhorn_iterations': 30, 'match_threshold': 0.3, 'show': False, 'ransac': False,
     'transform_type': "affine", 'device': "cuda:0"}
    params = {**default_params, **params}
    superglue_weights_path = params['superglue_weights_path']
    superglue = 'outdoor' # artifact
    sinkhorn_iterations = params['sinkhorn_iterations']
    match_threshold = params['match_threshold']
    show = params['show']
    device = params['device']
    transform_type = params['transform_type']
    ransac = params['ransac']

    config = {
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'descriptor_dim': 256,
        }       
    }

    model = sg.Matching(config).eval().to(device)
    model.superglue.load_state_dict(tc.load(superglue_weights_path))
 
    source_kp = [[item.pt[0], item.pt[1]] for item in source_keypoints]
    target_kp = [[item.pt[0], item.pt[1]] for item in target_keypoints]
    source_scores = tc.from_numpy(np.array([item.size for item in source_keypoints])).unsqueeze(0).to(device).type(tc.float32)
    target_scores = tc.from_numpy(np.array([item.size for item in target_keypoints])).unsqueeze(0).to(device).type(tc.float32)
    source_scores = source_scores / tc.sum(source_scores)
    target_scores = target_scores / tc.sum(target_scores)

    source_kp = tc.from_numpy(np.array(source_kp)).unsqueeze(0).to(device).type(tc.float32)
    target_kp = tc.from_numpy(np.array(target_kp)).unsqueeze(0).to(device).type(tc.float32)
    source_descriptors = np.repeat(source_descriptors, 2, axis=1)
    target_descriptors = np.repeat(target_descriptors, 2, axis=1)
    source_descriptors = tc.from_numpy(source_descriptors).permute(1, 0).unsqueeze(0).to(device).type(tc.float32)
    target_descriptors = tc.from_numpy(target_descriptors).permute(1, 0).unsqueeze(0).to(device).type(tc.float32)

    pred = model({'image0': resampled_source, 'image1': resampled_target,
     'keypoints0': source_kp, 'keypoints1': target_kp,
     'descriptors0': source_descriptors, 'descriptors1': target_descriptors,
     'scores0': source_scores, 'scores1': target_scores,
     })
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    matches = pred['matches0']

    source_kp = source_kp[0].detach().cpu().numpy()
    target_kp = target_kp[0].detach().cpu().numpy()
    valid = matches > -1
    mkpts0 = source_kp[valid]
    mkpts1 = target_kp[matches[valid]]
 
    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(resampled_source.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(source_kp[:, 0], source_kp[:, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(resampled_target.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(target_kp[:, 0], target_kp[:, 1], "r*")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(resampled_source.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.plot(mkpts0[:, 0], mkpts0[:, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(resampled_target.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
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
    
    final_transform = w.affine2theta(transform.astype(np.float64), (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
    if echo:
        print(f"Calculacted transform: {final_transform}")
    return final_transform

def descriptor_calculation(source : np.ndarray, target : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(4096)
    source_keypoints, source_descriptors = sift.detectAndCompute(source, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors
