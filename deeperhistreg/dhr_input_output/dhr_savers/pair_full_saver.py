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

from saver import WSISaver
from tiff_saver import TIFFSaver, default_params
from dhr_utils import utils as u

########################


class PairFullSaver():
    """
    TODO - documentation
    """
    def __init__(
        self,
        saver : WSISaver = TIFFSaver,
        save_params : dict = default_params,
        save_source_only : bool = False):
        """
        TODO
        """
        self.saver : WSISaver = saver()
        self.save_params = save_params
        self.save_source_only = save_source_only
        
    def unpad_images(
        self,
        source : Union[np.ndarray, tc.Tensor, pyvips.Image],
        target : Union[np.ndarray, tc.Tensor, pyvips.Image],
        initial_padding : Iterable[int] = None,
        unpad_with_target : bool = False,
        ) -> Tuple[Union[np.ndarray, tc.Tensor, pyvips.Image], Union[np.ndarray, tc.Tensor, pyvips.Image]]:
        """
        TODO
        """
        unpadded_source, unpadded_target = u.unpad(source, target, initial_padding, unpad_with_target)
        return unpadded_source, unpadded_target
    
    def crop_to_template(
        self,
        image : Union[np.ndarray, tc.Tensor, pyvips.Image],
        template_image : Union[np.ndarray, tc.Tensor, pyvips.Image],
        ) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        """
        TODO
        """
        cropped_image = u.crop_to_template(image, template_image)
        return cropped_image
        
    def save_images(
        self,
        source : Union[np.ndarray, tc.Tensor, pyvips.Image],
        target : Union[np.ndarray, tc.Tensor, pyvips.Image],
        source_path : Union[str, pathlib.Path],
        target_path : Union[str, pathlib.Path],
        initial_padding : Iterable[int] = None,
        to_template_shape : bool = False) -> None:
        """
        TODO
        """
        if to_template_shape:
            source_to_save = self.crop_to_template(source, target)
            target_to_save = target
            if initial_padding is not None:
                source_to_save, target_to_save = self.unpad_images(source_to_save, target_to_save, initial_padding, unpad_with_target = True)
        else:
            if initial_padding is not None:
                source_to_save, target_to_save = self.unpad_images(source, target, initial_padding, unpad_with_target = False)
            else:
                source_to_save = source
                target_to_save = target
        self.saver.save(source_to_save, source_path, self.save_params)
        if not self.save_source_only:
            self.saver.save(target_to_save, target_path, self.save_params)