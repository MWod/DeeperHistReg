# Ecosystem Imports
from abc import ABC, abstractmethod
from typing import Union, Iterable
from enum import Enum

# External Imports
import numpy as np
import torch as tc

############################################################

class LoadMode(Enum):
    PYTORCH = 1
    NUMPY = 2
    PYVIPS = 3
        


class WSILoader(ABC):
    """
    TODO - documentation
    """

    @abstractmethod
    def get_num_levels(self) -> int:
        """
        TODO - documentation
        """
        raise NotImplementedError()
        
    @abstractmethod    
    def get_resolutions(self) -> Iterable[int]:
        """
        TODO - documentation
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        """
        TODO - documentation
        """
        raise NotImplementedError()

    @abstractmethod
    def resample(self, resample_ratio : float) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        raise NotImplementedError()

    @abstractmethod
    def load_region(self, level : int, offset : tuple, shape : tuple) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        raise NotImplementedError()

    @abstractmethod
    def load_regions(self, level : int, offsets : Iterable[tuple], shape : tuple) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        raise NotImplementedError()

    @abstractmethod
    def load_level(self, level : int) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        raise NotImplementedError()

    