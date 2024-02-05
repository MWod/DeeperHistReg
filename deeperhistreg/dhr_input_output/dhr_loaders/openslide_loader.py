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

import pyvips

### Internal Imports ###

from loader import WSILoader, LoadMode
from dhr_utils import utils as u

########################


class OpenSlideLoader(WSILoader):
    """
    TODO - documentation
    """
    def __init__(
        self,
        image_path,
        mode=LoadMode.NUMPY):
        """
        
        """
        self.image_path = image_path
        self.mode = mode
        self.image = pyvips.Image.openslideload(self.image_path)
        self.num_levels = self.get_num_levels()
        self.resolutions = self.get_resolutions()
        
    def get_num_levels(self) -> int:
        """
        TODO - documentation
        """       
        return int(self.image.get('openslide.level-count'))
    
    def get_resolutions(self) -> Iterable[int]:
        """
        TODO - documentation
        """
        resolutions = []
        for level in range(self.num_levels):
            image = pyvips.Image.openslideload(self.image_path, level=level)
            height, width, bands = image.height, image.width, image.bands
            resolutions.append((height, width, bands))
        return resolutions

    def load(self) -> pyvips.Image:
        """
        TODO - documentation
        """
        return self.image

    def get_best_level(self, resample_ratio : float) -> pyvips.Image:
        """
        TODO - documentation
        """
        if self.num_levels == 1:
            return self.image, 0
        else:
            org_height, org_width = self.image.height, self.image.width
            desired_height, desired_width = org_height * resample_ratio, org_width * resample_ratio
            current_level = 0
            for i, resolution in enumerate(self.resolutions):
                if desired_height > resolution[0] or desired_width > resolution[1]:
                    break
                current_level = i
            return pyvips.Image.tiffload(self.image_path, page=current_level), current_level

    def update_resample_ratio(self, resample_ratio : float, level_to_use : int) -> float:
        """
        TODO - documentation
        """
        original_resolution = self.resolutions[0]
        updated_resolution = self.resolutions[level_to_use]
        resample_ratio =  (original_resolution[0] * resample_ratio) / updated_resolution[0]
        return resample_ratio

    def resample(self, resample_ratio : float) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        image, level_to_use = self.get_best_level(resample_ratio)
        if level_to_use > 0:
            resample_ratio = self.update_resample_ratio(resample_ratio, level_to_use)
        sigma = u.calculate_smoothing_sigma(resample_ratio)
        smoothed_image = image.gaussblur(sigma)
        resampled_image = smoothed_image.resize(resample_ratio, kernel='linear', vscale=resample_ratio)
        if self.mode == LoadMode.NUMPY:
            array = resampled_image.numpy()
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(resampled_image.numpy())
        elif self.mode == LoadMode.PYVIPS:
            array = resampled_image
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_region(self, level, offset, shape) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        image = pyvips.Image.openslideload(self.image_path, level=level)
        region = image.crop(offset[1], offset[0], shape[1], shape[0])
        if self.mode == LoadMode.NUMPY:
            array = region.numpy()
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(region.numpy())
        elif self.mode == LoadMode.PYVIPS:
            array = region[0:3]
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_regions(self, level, offsets, shape) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        image = pyvips.Image.openslideload(self.image_path, level=level)
        array = np.zeros((len(offsets), *shape, image.bands), dtype=np.uint8)
        for i, offset in enumerate(offsets):
            crop = image.crop(offset[1], offset[0], shape[1], shape[0])
            array[i, :, :, :] = crop.numpy()
        if self.mode == LoadMode.NUMPY:
            pass
        elif self.mode == LoadMode.PYTORCH:
            array = tc.from_numpy(array)
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_level(self, level) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        """
        TODO - documentation
        """
        image = pyvips.Image.openslideload(self.image_path, level=level)
        if self.mode == LoadMode.NUMPY:
            array = image.numpy()
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(image.numpy())
        elif self.mode == LoadMode.PYVIPS:
            array = image[0:3]
        else:
            raise ValueError("Unsupported mode.")
        return array

    