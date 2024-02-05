### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import logging

### External Imports ###
import numpy as np
import torch as tc

### Internal Imports ###
from loader import WSILoader, LoadMode
from dhr_utils import utils as u

########################


class SimpleLoader(WSILoader):
    """
    TODO - documentation
    """    
    def get_num_levels(self) -> int:
        """
        TODO - documentation
        """
        return 1
    
    def get_resolutions(self) -> Iterable[int]:
        """
        TODO - documentation
        """
        return [(self.image.shape[2], self.image.shape[3], self.image.shape[1])]

    def load(self) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if self.mode == LoadMode.NUMPY:
            return u.tensor_to_image(self.image)
        else:
            return self.image.clone()

    def resample(self, resample_ratio : float) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        sigma = u.calculate_smoothing_sigma(resample_ratio)
        smoothed_image = u.gaussian_smoothing(self.image.float(), sigma)
        resampled_image = u.resample(smoothed_image, 1 / resample_ratio).to(self.dtype)
        if self.mode == LoadMode.NUMPY:
            array = u.tensor_to_image(resampled_image)
        elif self.mode == LoadMode.PYTORCH:
            array = resampled_image
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_region(self, level : int, offset : tuple, shape : tuple) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        region = self.image[:, :, offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]]
        if self.mode == LoadMode.NUMPY:
            array = u.tensor_to_image(region)
        elif self.mode == LoadMode.PYTORCH:
            array = region
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_regions(self, level : int, offsets : Iterable[tuple], shape : tuple) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        array = tc.zeros((len(offsets), self.image.shape[1], *shape), dtype=self.image.dtype)
        for i, offset in enumerate(offsets):
            array[i, :, :, :] = self.image[0, :, offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]]
        if self.mode == LoadMode.NUMPY:
            array = u.tensor_to_image(array)
        elif self.mode == LoadMode.PYTORCH:
            array = array
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_level(self, level : int) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        if self.mode == LoadMode.NUMPY:
            array = u.tensor_to_image(self.image)
        elif self.mode == LoadMode.PYTORCH:
            array = self.image
        elif self.mode == LoadMode.PYVIPS:
            array = u.array_to_pyvips(u.tensor_to_image(self.image))
        else:
            raise ValueError("Unsupported mode.")
        return array

    