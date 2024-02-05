### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
import pathlib
import json

### External Imports ###
import torch as tc
import matplotlib.pyplot as plt

### Internal Imports ###
import dhr_input_output.input_output as io
from dhr_input_output.dhr_savers import saver as wsi_saver
from dhr_input_output.dhr_savers import displacement_saver
import dhr_utils.utils as u

########################

class ResultsSaver():
    def __init__(self,
                saver : wsi_saver.WSISaver,
                save_params : dict,
                save_path : Union[str, pathlib.Path],
                save_name : str,
                case_name : str,
                source : tc.Tensor=None,
                target : tc.Tensor=None,
                warped_source : tc.Tensor=None,
                displacement_field : tc.Tensor=None,
                postprocessing_params : dict=None,
                registration_time : float=None,
                extension : str =".jpg"):
        """
        TODO
        """
        self.saver = saver
        self.save_params = save_params
        self.save_path = save_path
        self.save_name = save_name
        self.case_name = case_name
        self.source = source
        self.target = target
        self.warped_source = warped_source
        self.displacement_field = displacement_field
        self.postprocessing_params = postprocessing_params
        self.registration_time = registration_time
        self.extension = extension

    def save(self):
        """
        Saves the current temporary results.
        """
        case_folder = pathlib.Path(self.save_path) / self.case_name / self.save_name
        if not case_folder.exists():
            case_folder.mkdir(parents=True, exist_ok=True)

        if self.source is not None:
            source_save_path = case_folder / ("source" + self.extension)
            self.saver.save(self.source, source_save_path, self.save_params)

        if self.target is not None:
            target_save_path = case_folder / ("target" + self.extension)
            self.saver.save(self.target, target_save_path, self.save_params)

        if self.warped_source is not None:
            warped_source_save_path = case_folder / ("warped_source" + self.extension)
            self.saver.save(self.warped_source, warped_source_save_path, self.save_params)

        if self.displacement_field is not None:
            displacement_field_save_path = str(case_folder / "displacement_field.mha")
            displacement_field_saver = displacement_saver.DisplacementFieldSaver()
            displacement_field_saver.save(self.displacement_field, displacement_field_save_path)

        if self.postprocessing_params is not None:
            postprocessing_save_path = str(case_folder / "postprocessing_params.json")
            with open(postprocessing_save_path, "w") as to_save:
                json.dump(self.postprocessing_params, to_save)

        if self.registration_time is not None:
            registration_time_save_path = str(case_folder / "registration_time.json")
            with open(registration_time_save_path, "w") as to_save:
                json.dump({"registration_time" : self.registration_time}, to_save)