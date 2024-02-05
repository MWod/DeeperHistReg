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


class DisplacementFieldLoader():
    """
    TODO - documentation
    """
    def __init__(self):
        pass

    def load(self, displacement_field_path : Union[pathlib.Path, str]) -> tc.Tensor:
        """
        TODO
        """
        displacement_field_np = sitk.GetArrayFromImage(sitk.ReadImage(str(displacement_field_path))).astype(np.float32)
        displacement_field_tc = u.np_df_to_tc_df(displacement_field_np)
        return displacement_field_tc