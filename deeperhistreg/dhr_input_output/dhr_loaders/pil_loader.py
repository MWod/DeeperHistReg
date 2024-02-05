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

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

### Internal Imports ###

import simple_loader
from loader import LoadMode
from dhr_utils import utils as u

########################

class PILLoader(simple_loader.SimpleLoader):
    """
    TODO - documentation
    """
    def __init__(self, image_path, mode=LoadMode.NUMPY):
        self.image_path = image_path
        self.mode = mode
        self.image = u.image_to_tensor(np.array(Image.open(self.image_path)))
        self.dtype = self.image.dtype
        self.num_levels = self.get_num_levels()
        self.resolutions = self.get_resolutions()