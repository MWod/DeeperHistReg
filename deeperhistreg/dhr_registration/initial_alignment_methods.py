import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import torch as tc

### Internal Imports ###
from dhr_utils import utils as u

from dhr_initial_alignment import exhaustive_rigid_search as ers
from dhr_initial_alignment import sift_ransac as sr
from dhr_initial_alignment import sift_superglue as ssg
from dhr_initial_alignment import superpoint_ransac as spr
from dhr_initial_alignment import superpoint_superglue as spsg
from dhr_initial_alignment import feature_combination as fc
from dhr_initial_alignment import io_affine as ioa
from dhr_initial_alignment import multi_feature as mf
########################

### Algorithms ###

# TODO - description of all the methods

def identity_initial_alignment(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return u.create_identity_transform(source)

def exhaustive_rigid_search(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ers.exhaustive_rigid_search(source, target, params)

def sift_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return sr.sift_ransac(source, target, params)

def sift_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return ssg.sift_superglue(source, target, params)

def superpoint_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return spr.superpoint_ransac(source, target, params)

def superpoint_superglue(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return spsg.superpoint_superglue(source, target, params)

def feature_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return fc.feature_based_combination(source, target, params)

def rotated_feature_combination(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return fc.rotated_feature_based_combination(source, target, params)

def multi_feature(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    return mf.multi_feature(source, target, params)

def instance_optimization_affine_registration(
    source : tc.Tensor,
    target : tc.Tensor,
    initial_displacement_field : Union[tc.Tensor, None],
    params : dict) -> tc.Tensor:
    return ioa.instance_optimization_affine_registration(source, target, initial_displacement_field, params)

def get_function(function_name : str) -> Callable:
    return getattr(current_file, function_name)