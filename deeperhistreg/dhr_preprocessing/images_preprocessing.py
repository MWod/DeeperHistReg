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

def initial_resampling(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
    """
    TODO - documentation
    """
    initial_resolution = params['initial_resolution']
    source_y_size, source_x_size, target_y_size, target_x_size = u.get_combined_size(source, target)
    source, target = u.normalize(source), u.normalize(target)
    resampling_ratio = u.calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), initial_resolution)
    initial_smoothing = max(resampling_ratio - 1, 0.1)
    smoothed_source, smoothed_target = u.gaussian_smoothing(source, initial_smoothing), u.gaussian_smoothing(target, initial_smoothing)
    preprocessed_source, preprocessed_target = u.resample(smoothed_source, resampling_ratio), u.resample(smoothed_target, resampling_ratio)
    postprocessing_params = {'resampling_ratio' : resampling_ratio}
    return preprocessed_source, preprocessed_target, postprocessing_params
