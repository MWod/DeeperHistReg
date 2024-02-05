import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
current_file = sys.modules[__name__]
from typing import Union, Iterable

### External Imports ###
import numpy as np
import torch as tc

### Internal Imports ###
import dhr_utils.utils as u

########################

def revert_initial_resampling(
    source : Union[np.ndarray, tc.Tensor],
    target : Union[np.ndarray, tc.Tensor],
    params : dict):
    """
    This function serves only for debugging purposes. It is not necessary to the processing pipeline.
    """
    initial_resolution = params['initial_resolution']
    resampling_ratio = params['resampling_ratio']
    postprocessed_source, postprocessed_target = u.resample(source, 1 / resampling_ratio), u.resample(target, 1 / resampling_ratio)
    # TODO - if necessary
    return postprocessed_source, postprocessed_target

def revert_initial_resampling_landmarks(landmarks : Union[np.ndarray, tc.Tensor], params : dict):
    """
    TODO - documentation
    """
    resampling_ratio = params['resampling_ratio']
    resampled_landmarks = landmarks * resampling_ratio
    return resampled_landmarks

def revert_basic_preprocessing_on_displacement_field(displacement_field : Union[np.ndarray, tc.Tensor], postprocessing_params : dict):
    """
    TODO - documentation
    """
    original_size = postprocessing_params['original_size']
    late_resample = postprocessing_params['late_resample']
    if late_resample:
        late_resample_ratio = postprocessing_params['late_resample_ratio']
        displacement_field = u.resample_displacement_field(displacement_field, 1 / late_resample_ratio)
    pad_to_same_size = postprocessing_params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = postprocessing_params['padding_params']
        displacement_field = u.unpad_displacement_field(displacement_field, padding_params)
    initial_resampling = postprocessing_params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = postprocessing_params['initial_resample_ratio']
        displacement_field = u.resample_displacement_field(displacement_field, 1 / initial_resample_ratio)
    current_size = displacement_field.shape[1:] if isinstance(displacement_field, np.ndarray) else (displacement_field.size(1), displacement_field.size(2))
    if current_size != original_size:
        print("Incorrect size during postprocessing.")
        displacement_field = u.resample_displacement_field_to_size(displacement_field, original_size)
    return displacement_field

def target_landmarks_postprocessing(target_landmarks : Union[np.ndarray, tc.Tensor], params : dict):
    """
    TODO - documentation
    """
    late_resample = params['late_resample']
    if late_resample:
        late_resample_ratio = params['late_resample_ratio']
        target_landmarks = target_landmarks * late_resample_ratio

    pad_to_same_size = params['pad_to_same_size']
    if pad_to_same_size:
        padding_params = params['padding_params']
        print(f"Padding params: {padding_params}")
        target_landmarks = u.unpad_landmarks(target_landmarks, padding_params['pad_1'])

    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resample_ratio = params['initial_resample_ratio']
        target_landmarks = target_landmarks * initial_resample_ratio
    return target_landmarks

def get_function(function_name):
    return getattr(current_file, function_name)