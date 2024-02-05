### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
import logging

### External Imports ###
import numpy as np
import torch as tc

import SimpleITK as sitk

### Internal Imports ###

from loader import LoadMode
import simple_loader
from dhr_utils import utils as u

########################


class SITKLoader(simple_loader.SimpleLoader):
    """
    TODO - documentation
    """
    def __init__(self, image_path, mode=LoadMode.NUMPY):
        self.image_path = image_path
        self.mode = mode
        self.image = u.image_to_tensor(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
        self.dtype = self.image.dtype
        self.num_levels = self.get_num_levels()
        self.resolutions = self.get_resolutions()