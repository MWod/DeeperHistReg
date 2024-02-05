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

def basic_preprocessing(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    source_landmarks : Union[tc.Tensor, np.ndarray],
    target_landmarks : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
    """
    TODO - documentation
    """
    postprocessing_params = dict()
    postprocessing_params['original_size'] = source.shape[0:2] if isinstance(source, np.ndarray) else (source.size(2), source.size(3))

    initial_resampling = params['initial_resampling']
    if initial_resampling:
        initial_resolution = params['initial_resolution']
        source_y_size, source_x_size, target_y_size, target_x_size = u.get_combined_size(source, target)
        initial_resample_ratio = u.calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), initial_resolution)
        initial_smoothing = max(initial_resample_ratio - 1, 0.1)
        source = u.resample(u.gaussian_smoothing(source, initial_smoothing), initial_resample_ratio)
        target = u.resample(u.gaussian_smoothing(target, initial_smoothing), initial_resample_ratio)
        postprocessing_params['initial_resample_ratio'] = initial_resample_ratio
        if source_landmarks is not None:
            source_landmarks = source_landmarks / initial_resample_ratio
        if target_landmarks is not None:
            target_landmarks = target_landmarks / initial_resample_ratio
    postprocessing_params['initial_resampling'] = initial_resampling 

    normalization = params['normalization']
    if normalization:
        source, target = u.normalize(source), u.normalize(target)

    convert_to_gray = params['convert_to_gray']
    if convert_to_gray:
        if params['flip_intensity']:
            source = 1 - u.convert_to_gray(source)
            target = 1 - u.convert_to_gray(target)
        else:
            source = u.convert_to_gray(source)
            target = u.convert_to_gray(target)

        clahe = params['clahe']
        if clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            src = clahe.apply((source[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            trg = clahe.apply((target[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
            source = tc.from_numpy((src.astype(np.float32) / 255)).to(source.device).unsqueeze(0).unsqueeze(0)
            target = tc.from_numpy((trg.astype(np.float32) / 255)).to(target.device).unsqueeze(0).unsqueeze(0)  

    return source, target, source_landmarks, target_landmarks, postprocessing_params