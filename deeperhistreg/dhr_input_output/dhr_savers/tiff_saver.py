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

### Internal Imports ###

from saver import WSISaver
from dhr_utils import utils as u

########################


"""
Possible compression: JPEG, DEFLATE, PACKBITS, CCITTFAX4, LZW, WEBP, ZSTD, JP2K
"""

default_params = {
    'compression' : 'lzw', #'deflate', # "lzw" "jpeg"
    'pyramid' : True,
    'bigtiff' : True,
    'strip' : False,
    # 'depth' : pyvips.enums.ForeignDzDepth.ONETILE, 
    # 'depth' : pyvips.enums.ForeignDzDepth.ONEPIXEL,
}

class TIFFSaver(WSISaver):
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
        
        NOTE - You can use any parameters defined for "tiffsave" in PyVips in in the save_params dictionary for customization.
        """
        save_params = {**default_params, **save_params}
        if isinstance(image, np.ndarray):
            to_save = pyvips.Image.new_from_array(image)
        if isinstance(image, tc.Tensor):
            to_save = pyvips.Image.new_from_array((u.tensor_to_image(image)))
        if isinstance(image, pyvips.Image):
            to_save = image
        to_save.tiffsave(save_path, **save_params)

