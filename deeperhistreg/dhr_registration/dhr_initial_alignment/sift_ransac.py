### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Tuple

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import cv2

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w
from dhr_building_blocks import cost_functions as cf

########################

def sift_ransac(
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
    try:
        source_points, target_points, matches = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors)
    except:
        final_transform = np.eye(3)
        final_transform = w.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
        if return_num_matches:
            return final_transform, 0
        else:
            return final_transform
    if echo:
        print(f"Number of source points: {len(source_points)}")
        print(f"Number of target points: {len(target_points)}")

    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(src, cmap='gray')
        plt.plot(source_points[:, 0, 0], source_points[:, 0, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(trg, cmap='gray')
        plt.plot(target_points[:, 0, 0], target_points[:, 0, 1], "r*")
        plt.show()
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    if show:
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(src,source_keypoints,trg,target_keypoints,matches,None,**draw_params)
        plt.imshow(img3)
        plt.show()
    try:
        transform, _ = cv2.estimateAffinePartial2D(source_points, target_points, 0) # [0, cv2.RANSAC, cv2.RHO, cv2.LMEDS]
        num_matches = len(source_points)
    except:
        transform = np.eye(3)[0:2, :]
        num_matches = 0
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

def descriptor_calculation(source : np.ndarray, target : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO
    """
    sift = cv2.SIFT_create(4096)
    source_keypoints, source_descriptors = sift.detectAndCompute(source, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors

def matcher(
    source_keypoints : np.ndarray,
    target_keypoints : np.ndarray,
    source_descriptors : np.ndarray,
    target_descriptors : np.ndarray,
    orb : bool=False) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    TODO
    """
    if not orb:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm = FLANN_INDEX_LSH,
            table_number = 12,
            key_size = 20,
            multi_probe_level = 2)
    search_params = dict(checks = 600)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    source_descriptors = source_descriptors.astype(np.float32)
    target_descriptors = target_descriptors.astype(np.float32)
    matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = []
    for m, n in matches:
       if m.distance < 0.7*n.distance:
           good_matches.append(m)
    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    return source_points, target_points, matches
