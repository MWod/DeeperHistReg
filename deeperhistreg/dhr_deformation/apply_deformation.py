### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union
from enum import Enum

### External Imports ###
import math
import numpy as np
import pyvips
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr

import pyvips


### Internal Imports ###

from dhr_utils import utils as u
from dhr_input_output.dhr_loaders import pair_full_loader as pair_loader
from dhr_input_output.dhr_savers import pair_full_saver as pair_saver
from dhr_input_output.dhr_loaders import tiff_loader as tiff_loader
from dhr_input_output.dhr_savers import tiff_saver as tiff_saver
from dhr_input_output.dhr_loaders import displacement_loader as df_loader

from dhr_utils import warping as w


########################

class WarpingMode(Enum):
    PYVIPS = 0
    PYTORCH = 1

def apply_deformation(
    source_image_path : Union[pathlib.Path, str],
    target_image_path : Union[pathlib.Path, str],
    warped_image_path : Union[pathlib.Path, str],
    displacement_field_path : Union[pathlib.Path, str],
    warping_mode : WarpingMode = WarpingMode.PYVIPS,
    loader : tiff_loader.WSILoader = tiff_loader.WSILoader,
    saver : tiff_saver.WSISaver = tiff_saver.WSISaver,
    save_params : dict = tiff_saver.default_params,
    level : int = 0,
    pad_value : float = 255.0,
    save_source_only : bool = True,
    to_template_shape : bool = True,
    to_save_target_path : Union[pathlib.Path, str] = None) -> None:
    """
    TODO - documentation
    Compatible pairs:
    - Loader: tiff, Warping: all: Saver: tiff
    - Loader: openslide, : Warping : all, Saver: tiff
    - Loader: vips : Warping : all, Saver : tiff
    - Loader: sitk : Warping : torch, Saver: tiff, sitk, PIL
    - Loader: PIL : Warping : torch Saver: tiff, sitk, PIL
    """
    if warping_mode == WarpingMode.PYVIPS:
        apply_deformation_pyvips(source_image_path, target_image_path, warped_image_path, displacement_field_path, 
                                 loader, saver, save_params, level, pad_value, save_source_only, to_template_shape, to_save_target_path)
    elif warping_mode == WarpingMode.PYTORCH:
        apply_deformation_pytorch(source_image_path, target_image_path, warped_image_path, displacement_field_path, 
                                 loader, saver, save_params, level, pad_value, save_source_only, to_template_shape, to_save_target_path)
    else:
        raise ValueError("Unsupported mode.")

def apply_deformation_pyvips(
    source_image_path : Union[pathlib.Path, str],
    target_image_path : Union[pathlib.Path, str],
    warped_image_path : Union[pathlib.Path, str],
    displacement_field_path : Union[pathlib.Path, str],
    loader : tiff_loader.WSILoader = tiff_loader.WSILoader,
    saver : tiff_saver.WSISaver = tiff_saver.WSISaver,
    save_params : dict = tiff_saver.default_params,
    level : int = 0,
    pad_value : float = 255.0,
    save_source_only : bool = True,
    to_template_shape : bool = True,
    to_save_target_path : Union[pathlib.Path, str] = None) -> None:
    """
    TODO - documentation
    """ 
    mode = tiff_loader.LoadMode.PYVIPS
    source, target, padding_params = pair_loader.PairFullLoader(source_image_path, target_image_path, loader, mode).load_image(level=level, pad_value=pad_value)
    displacement_field = df_loader.DisplacementFieldLoader().load(displacement_field_path)
    warped_source = w.warp_pyvips_with_tc_df(source, displacement_field)
    to_save = pair_saver.PairFullSaver(saver, save_params, save_source_only=save_source_only)
    to_save.save_images(warped_source, target, warped_image_path, to_save_target_path, initial_padding=padding_params, to_template_shape=to_template_shape)


def apply_deformation_pytorch(
    source_image_path : Union[pathlib.Path, str],
    target_image_path : Union[pathlib.Path, str],
    warped_image_path : Union[pathlib.Path, str],
    displacement_field_path : Union[pathlib.Path, str],
    loader : tiff_loader.WSILoader = tiff_loader.WSILoader,
    saver : tiff_saver.WSISaver = tiff_saver.WSISaver,
    save_params : dict = tiff_saver.default_params,
    level : int = 0,
    pad_value : float = 255.0,
    save_source_only : bool = True,
    to_template_shape : bool = True,
    to_save_target_path : Union[pathlib.Path, str] = None) -> None:
    """
    TODO - documentation
    """
    mode = tiff_loader.LoadMode.PYTORCH
    source, target, padding_params = pair_loader.PairFullLoader(source_image_path, target_image_path, loader, mode).load_image(level=level, pad_value=pad_value)
    displacement_field = df_loader.DisplacementFieldLoader().load(displacement_field_path)
    source = u.image_to_tensor(source.numpy()).to(tc.float32)
    target = u.image_to_tensor(target.numpy()).to(tc.float32)
    with tc.set_grad_enabled(False):
        displacement_field = u.resample_displacement_field_to_size(displacement_field, source.shape[2:])
        warped_source = w.warp_tensor(source, displacement_field)
    warped_source = warped_source.cpu().to(tc.uint8)
    to_save = pair_saver.PairFullSaver(saver, save_params, save_source_only=save_source_only)
    to_save.save_images(warped_source, target, warped_image_path, to_save_target_path, initial_padding=padding_params, to_template_shape=to_template_shape)