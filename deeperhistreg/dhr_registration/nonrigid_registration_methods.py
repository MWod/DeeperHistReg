import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union

### External Imports ###
import torch as tc

### Internal Imports ###
from dhr_nonrigid_registration import io_bsplines as iob
from dhr_nonrigid_registration import io_nonrigid as ion
from dhr_utils import utils as u

########################

def identity_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return u.create_identity_displacement_field(source)

def residual_unet_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    raise NotImplementedError # TODO

def lapirn_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    raise NotImplementedError # TODO

def transformer_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    raise NotImplementedError # TODO

def instance_optimization_nonrigid_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ion.instance_optimization_nonrigid_registration(source, target, initial_displacement_field, params)

def instance_optimization_nonrigid_registration_lbfgs(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ion.instance_optimization_nonrigid_registration_lbfgs(source, target, initial_displacement_field, params)

def instance_optimization_bsplines_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return iob.instance_optimization_bsplines_registration(source, target, initial_displacement_field, params)


def get_function(function_name):
    return getattr(current_file, function_name)






