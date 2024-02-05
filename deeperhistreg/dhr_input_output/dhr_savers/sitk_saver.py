### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import logging
import pathlib

### External Imports ###
import numpy as np
import torch as tc
import pyvips
import SimpleITK as sitk

### Internal Imports ###

from saver import WSISaver
from dhr_utils import utils as u

########################

default_params = {
}

class SITKSaver(WSISaver):
    """
    TODO - documentation
    """
    def __init__(self):
        pass
    
    def save(
        self,
        image : Union[np.ndarray, tc.Tensor, pyvips.Image],
        save_path : Union[str, pathlib.Path],
        save_params : dict) -> None:
        """
        TODO - documentation
        """
        save_params = {**default_params, **save_params}
        if isinstance(image, np.ndarray):
            to_save = image
        if isinstance(image, tc.Tensor):
            to_save = u.tensor_to_image(image)
        if isinstance(image, pyvips.Image):
            to_save = image.numpy()
        if to_save.shape[2] == 1:
            to_save = to_save.repeat(3, axis=2)
        to_save = sitk.GetImageFromArray(to_save)
        sitk.WriteImage(to_save, str(save_path))

