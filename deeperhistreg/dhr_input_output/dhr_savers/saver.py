# Ecosystem Imports
from abc import ABC, abstractmethod
from typing import Union, Iterable
import pathlib

# External Imports
import numpy as np
import torch as tc
import pyvips

############################################################


class WSISaver(ABC):
    """
    TODO - documentation
    """
    @abstractmethod
    def save(
        self,
        image : Union[np.ndarray, tc.Tensor, pyvips.Image],
        save_path : Union[str, pathlib.Path],
        save_params : dict) -> None:
        """
        TODO - documentation
        """
        raise NotImplementedError()


    