### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Sequence, Tuple
current_file = sys.modules[__name__]

### External Imports ###
import numpy as np
import torch as tc
import cv2


### Internal Imports ###
from dhr_utils import utils as u

########################


def initial_resampling_landmarks(landmarks : Union[tc.Tensor, np.ndarray], params : dict) -> Union[tc.Tensor, np.ndarray]:
    """
    TODO - documentation
    """
    resampling_ratio = params['resampling_ratio']
    resampled_landmarks = landmarks / resampling_ratio
    return resampled_landmarks

def landmarks_preprocessing(
    source_landmarks : Union[tc.Tensor, np.ndarray],
    target_landmarks : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray]]:
    """
    TODO - documentation
    """
    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = params['initial_resample_ratio']
        source_landmarks = source_landmarks / initial_resample_ratio
        target_landmarks = target_landmarks / initial_resample_ratio

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = params['padding_params']
        source_landmarks = u.pad_landmarks(source_landmarks, padding_params['pad_1'])
        target_landmarks = u.pad_landmarks(target_landmarks, padding_params['pad_2'])

    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        source_landmarks = source_landmarks / late_resample_ratio
        target_landmarks = target_landmarks / late_resample_ratio
    return source_landmarks, target_landmarks

def target_landmarks_preprocessing(target_landmarks : Union[tc.Tensor, np.ndarray], params : dict) -> Union[tc.Tensor, np.ndarray]:
    """
    TODO - documentation
    """
    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = params['initial_resample_ratio']
        target_landmarks = target_landmarks / initial_resample_ratio

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = params['padding_params']
        target_landmarks = u.pad_landmarks(target_landmarks, padding_params['pad_2'])

    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        target_landmarks = target_landmarks / late_resample_ratio
    return target_landmarks

