### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import SimpleITK as sitk
import PIL
import matplotlib.pyplot as plt


import openslide

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w

########################


def load_slide(load_path, level, load_slide=False):
    slide = openslide.OpenSlide(load_path)
    dimension = slide.level_dimensions[level]
    image = slide.read_region((0, 0), level, dimension)
    image = np.asarray(image)[:, :, 0:3].astype(np.float32)
    # image = u.normalize(image)
    if load_slide:
        return image, slide
    else:
        return image

def load_image(load_path, mode=None):
    # TODO - documentation
    if mode is None:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(load_path))).astype(np.float32)
        if len(array.shape) == 3:
            if array.shape[2] == 4:
                array = array[:, :, 0:3]
    elif mode == "ANHIR":
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(load_path))).astype(np.float32)
    elif mode == "ACROBAT":
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(load_path))).astype(np.float32)[:, :, 0:3]
    elif mode == "PYRAMID":
        array = openslide.OpenSlide(load_path)
    return array

def save_image(image, save_path, renormalize=True):
    # TODO - documentation
    if not save_path.parents[0].exists():
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
    extension = os.path.splitext(save_path)[1]
    if image.shape[2] == 3:
        if extension == ".jpg" or extension == ".jpeg" or extension == ".png":
            if renormalize:
                image = (image * 255)
                image = image.astype(np.uint8)
                image = PIL.Image.fromarray(image)
                image.save(str(save_path))
            else:
                image = image.astype(np.uint8)
                image = PIL.Image.fromarray(image)
                image.save(str(save_path))
        else:
            sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    elif image.shape[2] == 1:
        if renormalize:
            image = (image[:, :, 0]*255)
            image = image.astype(np.uint8)
        else:
            image = (image[:, :, 0])
            image = image.astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    else:
        raise ValueError("Unsupported image format.")

def load_displacement_field(load_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(load_path))).astype(np.float32)

def save_displacement_field(displacement_field, save_path):
    sitk.WriteImage(sitk.GetImageFromArray(displacement_field.astype(np.float32)), str(save_path), useCompression=True)