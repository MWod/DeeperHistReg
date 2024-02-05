### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
current_file = sys.modules[__name__]
from typing import Union, Callable, Iterable, Sequence, Tuple

### External Imports ###
import numpy as np
import torch as tc

### Internal Imports ###
import general_preprocessing as gp
import images_preprocessing as ip
import landmarks_preprocessing as lp

########################

### General Preprocessing ###

def basic_preprocessing(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    source_landmarks : Union[tc.Tensor, np.ndarray],
    target_landmarks : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
    return gp.basic_preprocessing(source, target, source_landmarks, target_landmarks, params)

### Images Preprocessing ###

def initial_resampling(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
    return ip.initial_resampling(source, target, params)

### Landmarks Preprocessing ###

def initial_resampling_landmarks(landmarks : Union[tc.Tensor, np.ndarray], params : dict) -> Union[tc.Tensor, np.ndarray]:
    return lp.initial_resampling_landmarks(landmarks, params)

def landmarks_preprocessing(
    source_landmarks : Union[tc.Tensor, np.ndarray],
    target_landmarks : Union[tc.Tensor, np.ndarray],
    params : dict) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray]]:
    return lp.landmarks_preprocessing(source_landmarks, target_landmarks, params)

def target_landmarks_preprocessing(target_landmarks : Union[tc.Tensor, np.ndarray], params : dict) -> Union[tc.Tensor, np.ndarray]:
    return lp.target_landmarks_preprocessing(target_landmarks, params)


### Utilities ###

def get_function(function_name) -> Callable:
    return getattr(current_file, function_name)