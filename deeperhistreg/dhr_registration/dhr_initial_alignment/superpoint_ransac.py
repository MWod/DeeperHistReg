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
from dhr_networks import superpoint as sp

########################


def superpoint_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    echo = params['echo']
    resolution = params['registration_size']
    show = params['show']
    try:
        return_num_matches = params['return_num_matches']
    except KeyError:
        return_num_matches = False

    ### Initial Resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)   
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    ### SuperPoint keypoints and descriptors ###
    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg  = u.tensor_to_image(resampled_target)[:, :, 0]
    try:
        source_keypoints, _, target_keypoints, _, num_matches = calculate_keypoints(src, trg, params)
    except:
        final_transform = np.eye(3)
        final_transform = w.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
        num_matches = 0
        return final_transform

    if echo:
        print(f"Number of source keypoints: {len(source_keypoints)}")
        print(f"Number of target keypoints: {len(target_keypoints)}")

    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(src, cmap='gray')
        plt.plot(source_keypoints[:, 0], source_keypoints[:, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(trg, cmap='gray')
        plt.plot(target_keypoints[:, 0], target_keypoints[:, 1], "r*")
        plt.show()

    try:
        # [0, cv2.RANSAC, cv2.RHO, cv2.LMEDS]
        transform, _ = cv2.estimateAffinePartial2D(source_keypoints, target_keypoints, 1)
    except:
        transform = np.eye(3)[0:2, :]

    final_transform = np.eye(3)
    final_transform[0:2, 0:3] = transform
    try:
        final_transform = np.linalg.inv(final_transform)
    except:
        final_transform = np.eye(3)
    final_transform = w.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
    if echo:
        print(f"Calculacted transform: {final_transform}")
        print(f"Number of matches: {num_matches}")
    if return_num_matches:
        return final_transform, num_matches
    else:
        return final_transform

def calculate_keypoints(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> Tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
    """
    TODO
    """
    ### Params Unpack ###
    default_params = {'weights_path': p.superpoint_model_path, 'nms_dist': 4, 'conf_thresh': 0.015, "nn_thresh": 0.7, 'cuda': True, 'show': False}
    params = {**default_params, **params}
    weights_path = params['weights_path']
    nms_dist = params['nms_dist']
    conf_thresh = params['conf_thresh']
    nn_thresh = params['nn_thresh']
    cuda = params['cuda']
    show = params['show']

    ### Model Creation ###
    model = sp.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda=cuda)

    ### Keypoints / Descriptors Calculation ###
    src_pts, src_desc, src_heatmap = model.run(source)
    trg_pts, trg_desc, trg_heatmap = model.run(target)

    if show:
        plt.figure()
        plt.imshow(source, cmap='gray')
        plt.plot(src_pts[0, :], src_pts[1, :], "r*")

        plt.figure()
        plt.imshow(target, cmap='gray')
        plt.plot(trg_pts[0, :], trg_pts[1, :], "r*")

    matches = sp.nn_match_two_way(src_desc, trg_desc, nn_thresh)

    if show:
        plt.figure()
        plt.imshow(source, cmap='gray')
        plt.plot(src_pts[0, matches[0, :].astype(np.int32)], src_pts[1, matches[0, :].astype(np.int32)], "r*")

        plt.figure()
        plt.imshow(target, cmap='gray')
        plt.plot(trg_pts[0, matches[1, :].astype(np.int32)], trg_pts[1, matches[1, :].astype(np.int32)], "r*")

    src_pts = src_pts[:, matches[0, :].astype(np.int32)].swapaxes(0, 1)[:, 0:2].astype(np.float32)
    trg_pts = trg_pts[:, matches[1, :].astype(np.int32)].swapaxes(0, 1)[:, 0:2].astype(np.float32)
    src_desc = src_desc[:, matches[0, :].astype(np.int32)].swapaxes(0, 1)
    trg_desc = trg_desc[:, matches[1, :].astype(np.int32)].swapaxes(0, 1)
    return src_pts, src_desc, trg_pts, trg_desc, len(matches[0])
