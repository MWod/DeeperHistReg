### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import pathlib

### External Imports ###
import numpy as np
import torch as tc
import SimpleITK as sitk


### Internal Imports ###

from dhr_utils import utils as u

########################

class DisplacementFieldSaver():
    """
    TODO - documentation
    """
    def __init__(self):
        pass

    def save(self, displacement_field : Union[tc.Tensor, np.ndarray], displacement_field_path : Union[pathlib.Path, str]) -> None:
        """
        TODO
        """
        if isinstance(displacement_field, tc.Tensor):
            displacement_field = u.tc_df_to_np_df(displacement_field)
        elif isinstance(displacement_field, np.ndarray):
            pass
        else:
            raise ValueError("Displacement field must be NumPy ndarray or PyTorch Tensor.")
        sitk.WriteImage(sitk.GetImageFromArray(displacement_field.astype(np.float32)), str(displacement_field_path))