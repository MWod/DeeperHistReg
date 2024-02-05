### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Tuple
import logging

### External Imports ###
import numpy as np
import torch as tc

import pyvips

### Internal Imports ###

from loader import WSILoader, LoadMode
from vips_loader import VIPSLoader
from dhr_utils import utils as u

########################


class PairFullLoader():
    """
    TODO - documentation
    """
    def __init__(
        self,
        source_path : Union[str, pathlib.Path],
        target_path : Union[str, pathlib.Path],
        loader : WSILoader = VIPSLoader,
        mode : LoadMode = LoadMode.NUMPY,
        ):
        """
        TODO
        """
        self.source_path = source_path
        self.target_path = target_path
        self.mode = mode
        self.source_loader : WSILoader = loader(self.source_path, mode=self.mode)
        self.target_loader : WSILoader = loader(self.target_path, mode=self.mode)
    
    def pad_to_same_shape(
        self,
        source : Union[tc.Tensor, np.ndarray],
        target : Union[tc.Tensor, np.ndarray],
        pad_value : float) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
        """
        TODO
        """
        padded_source, padded_target, padding_params = u.pad_to_same_size(source, target, pad_value)
        return padded_source, padded_target, padding_params
    
    def pad_to_same_shape_image(
        self,
        source : pyvips.Image,
        target : pyvips.Image,
        pad_value : float) -> Tuple[pyvips.Image, pyvips.Image, dict]:
        """
        TODO
        """
        padded_source, padded_target, padding_params = u.pad_to_same_size(source, target, pad_value)
        return padded_source, padded_target, padding_params
        
    def load_array(
        self,
        source_resample_ratio : float = 1.0,
        target_resample_ratio : float = 1.0,
        pad_value : float = 255) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray], dict]:
        """
        TODO
        """
        source = self.source_loader.resample(source_resample_ratio)
        target = self.target_loader.resample(target_resample_ratio)
        padded_source, padded_target, padding_params = self.pad_to_same_shape(source, target, pad_value)
        return padded_source, padded_target, padding_params
    
    def load_image(
        self,
        level : int = 0,
        pad_value : float = 255) -> Tuple[Union[pyvips.Image, pyvips.Image], dict]:
        """
        TODO
        """
        self.source_loader.mode = LoadMode.PYVIPS
        self.target_loader.mode = LoadMode.PYVIPS
        source = self.source_loader.load_level(level=level)
        target = self.target_loader.load_level(level=level)
        self.source_loader.mode = self.mode
        self.target_loader.mode = self.mode
        padded_source, padded_target, padding_params = self.pad_to_same_shape_image(source, target, pad_value)
        return padded_source, padded_target, padding_params        