### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable
import logging
import time
import pathlib

### External Imports ###
import numpy as np
import torch as tc

### Internal Imports ###
from dhr_preprocessing import preprocessing as pre
from dhr_registration import initial_alignment_methods as ia
from dhr_registration import nonrigid_registration_methods as nr
from dhr_deformation import apply_deformation as adf
from dhr_utils import utils as u
from dhr_utils import warping as w

from dhr_input_output.dhr_loaders import tiff_loader
from dhr_input_output.dhr_loaders import vips_loader
from dhr_input_output.dhr_loaders import pil_loader
from dhr_input_output.dhr_loaders import openslide_loader
from dhr_input_output.dhr_loaders import pair_full_loader

from dhr_input_output.dhr_savers import pil_saver
from dhr_input_output.dhr_savers import tiff_saver
from dhr_input_output.dhr_savers import results_saver as rs

########################

loader_mapper = {
    'tiff' : tiff_loader.TIFFLoader,
    'vips' : vips_loader.VIPSLoader,
    'pil' : pil_loader.PILLoader,
    'openslide' : openslide_loader.OpenSlideLoader
}
    
saver_mapper = {
    'tiff' : tiff_saver.TIFFSaver,
    'pil' : pil_saver.PILSaver,
}

saver_params_mapper = {
    'tiff' : tiff_saver.default_params,
    'pil' : pil_saver.default_params,
}

class DeeperHistReg_FullResolution():
    def __init__(self, registration_parameters : dict):
        """
        TODO
        """
        self.registration_parameters = registration_parameters
        self.registration_parameters['device'] = self.registration_parameters['device'] if tc.cuda.is_available() else "cpu"
        self.device = self.registration_parameters['device']
        self.echo = self.registration_parameters['echo']
        self.case_name = self.registration_parameters['case_name']
        self.logging_path = self.registration_parameters['logging_path']
        if self.logging_path is not None:
            self.logger = logging.getLogger(self.case_name)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.FileHandler(self.logging_path))

    def load_images(self) -> None:
        """
        TODO
        """
        loading_params = self.registration_parameters['loading_params']
        pad_value = loading_params['pad_value']
        loader = loader_mapper[loading_params['loader']]
        source_resample_ratio = loading_params['source_resample_ratio']
        target_resample_ratio = loading_params['target_resample_ratio']
        b_t = time.time()
        pair_loader = pair_full_loader.PairFullLoader(self.source_path, self.target_path, loader=loader, mode=pair_full_loader.LoadMode.PYTORCH)
        self.source, self.target, self.padding_params = pair_loader.load_array(source_resample_ratio=source_resample_ratio,
                                                                                target_resample_ratio=target_resample_ratio,
                                                                                pad_value=pad_value)
        self.padding_params['source_resample_ratio'] = source_resample_ratio
        self.padding_params['target_resample_ratio'] = target_resample_ratio
        
        e_t = time.time()
        self.org_source, self.org_target = self.source.to(tc.float32).to(self.device), self.target.to(tc.float32).to(self.device)
        if self.logging_path is not None:
            self.logger.info(f"Image loading finished.")
            self.logger.info(f"Source shape: {self.source.shape}")
            self.logger.info(f"Target shape: {self.target.shape}")
            self.logger.info(f"Loading time: {e_t - b_t} seconds.")

    def run_prepreprocessing(self) -> None:
        """
        TODO
        """
        with tc.set_grad_enabled(False):
            b_t = time.time()
            self.preprocessing_params = self.registration_parameters['preprocessing_params']
            preprocessing_function = pre.get_function(self.preprocessing_params['preprocessing_function'])
            self.pre_source, self.pre_target, _, _, self.postprocessing_params = preprocessing_function(self.org_source, self.org_target, None, None, self.preprocessing_params)
            e_t = time.time()
            self.preprocessing_time = e_t - b_t
            self.padding_params['initial_resampling'] = self.postprocessing_params['initial_resampling']
            if self.postprocessing_params['initial_resampling']:
                self.padding_params['initial_resample_ratio'] = self.postprocessing_params['initial_resample_ratio']
            if self.logging_path is not None:
                self.logger.info(f"Preprocessing finished.")
                self.logger.info(f"Preprocessed source shape: {self.pre_source.shape}")
                self.logger.info(f"Preprocessed target shape: {self.pre_target.shape}")
                self.logger.info(f"Preprocessing time: {self.preprocessing_time} seconds.")
            self.current_displacement_field = u.create_identity_displacement_field(self.pre_source)
            tc.cuda.empty_cache()

    def save_preprocessing(self) -> None:
        """
        TODO
        """
        with tc.set_grad_enabled(False):
            save_preprocessing = self.registration_parameters['preprocessing_params']['save_results']
            if save_preprocessing:
                save_name = "Preprocessing"
                rs.ResultsSaver(
                    saver_mapper[self.registration_parameters['saving_params']['saver']](),
                    saver_params_mapper[self.registration_parameters['saving_params']['save_params']],
                    self.save_path,
                    save_name,
                    self.case_name,
                    source = u.normalize_to_window(u.tensor_to_image(self.pre_source), 0, 255).astype(np.uint8),
                    warped_source=None,
                    target = u.normalize_to_window(u.tensor_to_image(self.pre_target), 0, 255).astype(np.uint8),
                    displacement_field=None,
                    postprocessing_params=self.postprocessing_params,
                    registration_time=self.preprocessing_time,
                    extension=self.registration_parameters['saving_params']['extension']).save()
                if self.logging_path is not None:
                    self.logger.info(f"Preprocessing saved.")

    def run_initial_registration(self) -> None:
        """
        TODO
        """
        if self.registration_parameters['run_initial_registration']:
            b_t = time.time()
            initial_registration_params = self.registration_parameters['initial_registration_params']
            initial_registration_function = ia.get_function(initial_registration_params['initial_registration_function'])
            self.initial_transform = initial_registration_function(self.pre_source, self.pre_target, initial_registration_params)
            self.initial_displacement_field = w.tc_transform_to_tc_df(self.initial_transform, self.pre_source.size())
            e_t = time.time()
            self.initial_registration_time = e_t - b_t
            if self.logging_path is not None:
                self.logger.info(f"Initial registration finished.")
                self.logger.info(f"Initial registration time: {self.initial_registration_time} seconds.")
            self.current_displacement_field = self.initial_displacement_field
            tc.cuda.empty_cache()

    def save_initial_registration(self) -> None:
        """
        TODO
        """
        with tc.set_grad_enabled(False):
            save_initial_registration = self.registration_parameters['initial_registration_params']['save_results']
            if save_initial_registration:
                save_name = "Initial_Registration"
                warped_source = w.warp_tensor(self.pre_source, self.initial_displacement_field)
                rs.ResultsSaver(
                    saver_mapper[self.registration_parameters['saving_params']['saver']](),
                    saver_params_mapper[self.registration_parameters['saving_params']['save_params']],
                    self.save_path,
                    save_name,
                    self.case_name,
                    source = u.normalize_to_window(u.tensor_to_image(self.pre_source), 0, 255).astype(np.uint8),
                    warped_source = u.normalize_to_window(u.tensor_to_image(warped_source), 0, 255).astype(np.uint8),
                    target = u.normalize_to_window(u.tensor_to_image(self.pre_target), 0, 255).astype(np.uint8),
                    displacement_field=self.initial_displacement_field,
                    registration_time=self.initial_registration_time,
                    extension=self.registration_parameters['saving_params']['extension']).save()
                if self.logging_path is not None:
                    self.logger.info(f"Initial alignment saved.")

    def run_nonrigid_registration(self) -> None:
        """
        TODO
        """
        if self.registration_parameters['run_nonrigid_registration']:
            b_t = time.time()
            nonrigid_registration_params = self.registration_parameters['nonrigid_registration_params']
            nonrigid_registration_function = nr.get_function(nonrigid_registration_params['nonrigid_registration_function'])
            self.nonrigid_displacement_field = nonrigid_registration_function(self.pre_source, self.pre_target, self.current_displacement_field, nonrigid_registration_params)
            e_t = time.time()
            self.nonrigid_registration_time = e_t - b_t
            if self.logging_path is not None:
                self.logger.info(f"Nonrigid registration finished.")
                self.logger.info(f"Nonrigid registration time: {self.nonrigid_registration_time} seconds.")
            self.current_displacement_field = self.nonrigid_displacement_field
            tc.cuda.empty_cache()

    def save_nonrigid_registration(self) -> None:
        """
        TODO
        """
        with tc.set_grad_enabled(False):
            save_nonrigid_registration = self.registration_parameters['nonrigid_registration_params']['save_results']
            if save_nonrigid_registration:
                save_name = "Nonrigid_Registration"
                warped_source = w.warp_tensor(self.pre_source, self.nonrigid_displacement_field)
                rs.ResultsSaver(
                    saver_mapper[self.registration_parameters['saving_params']['saver']](),
                    saver_params_mapper[self.registration_parameters['saving_params']['save_params']],
                    self.save_path,
                    save_name,
                    self.case_name,
                    source = u.normalize_to_window(u.tensor_to_image(self.pre_source), 0, 255).astype(np.uint8),
                    warped_source = u.normalize_to_window(u.tensor_to_image(warped_source), 0, 255).astype(np.uint8),
                    target = u.normalize_to_window(u.tensor_to_image(self.pre_target), 0, 255).astype(np.uint8),
                    displacement_field=self.nonrigid_displacement_field,
                    registration_time=self.nonrigid_registration_time,
                    extension = self.registration_parameters['saving_params']['extension']).save()
                if self.logging_path is not None:
                    self.logger.info(f"Nonrigid registration saved.")

    def preprocessing(self) -> None:
        """
        TODO
        """
        self.run_prepreprocessing()
        self.save_preprocessing()

    def initial_registration(self) -> None:
        """
        TODO
        """
        self.run_initial_registration()
        self.save_initial_registration()

    def nonrigid_registration(self) -> None:
        """
        TODO
        """
        self.run_nonrigid_registration()
        self.save_nonrigid_registration()

    def save_final(self) -> None:
        """
        TODO
        """
        save_final_df = self.registration_parameters['save_final_displacement_field']
        if save_final_df:
            saver = None
            save_params = None
            save_name = "Results_Final"
            rs.ResultsSaver(
                            saver,
                            save_params,
                            self.save_path,
                            save_name,
                            self.case_name,
                            source=None,
                            warped_source=None,
                            target=None,
                            displacement_field=self.current_displacement_field,
                            postprocessing_params=self.padding_params,
                            registration_time=self.total_registration_time,
                            extension=None).save()
            if self.logging_path is not None:
                self.logger.info(f"Final displacement field saved.")

        save_final = self.registration_parameters['save_final_images']
        if save_final and save_final_df:
            displacement_field_path = pathlib.Path(self.save_path) / self.case_name / save_name / "displacement_field.mha"
            adf.apply_deformation_pyvips(
                self.source_path,
                self.target_path,
                pathlib.Path(self.save_path) / self.case_name / save_name / str("warped_source" + self.registration_parameters['saving_params']['final_extension']),
                displacement_field_path,
                loader = loader_mapper[self.registration_parameters['loading_params']['loader']],
                saver = saver_mapper[self.registration_parameters['saving_params']['final_saver']],
                level = self.registration_parameters['loading_params']['final_level'],
                pad_value = self.registration_parameters['loading_params']['pad_value'],
                save_source_only = True,
                to_template_shape = True)
            if self.logging_path is not None:
                self.logger.info(f"Final images saved.")


    def run_registration(
        self, source_path : str,
        target_path : str,
        save_path : str) -> None:
        """
        TODO
        """
        self.source_path, self.target_path, self.save_path = source_path, target_path, save_path
        b_t = time.time()
        self.load_images()
        self.preprocessing()
        self.initial_registration()
        self.nonrigid_registration()
        e_t = time.time()
        self.total_registration_time = e_t - b_t
        if self.logging_path is not None:
            self.logger.info(f"Total registration time: {self.total_registration_time} seconds.")
        self.save_final()