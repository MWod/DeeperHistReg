### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
import json


### External Imports ###
import numpy as np

### Internal Imports ###

########################


def create_identity():
    params = dict()
    params['device'] = "cpu"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Identity"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 2048
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = False
    preprocessing_params['save_results'] = True
    params['preprocessing_params'] = preprocessing_params
    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "identity_initial_alignment"
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params
    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "identity_nonrigid_registration"
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params



def default_initial():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 2048
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "multi_feature"
    initial_registration_params['registration_size'] = 620
    initial_registration_params['registration_sizes'] = [150, 200, 250, 300, 350, 400, 450, 500]
    initial_registration_params['transform_type'] = 'rigid'
    initial_registration_params['keypoint_threshold'] = 0.005
    initial_registration_params['match_threshold'] = 0.3
    initial_registration_params['sinkhorn_iterations'] = 50
    initial_registration_params['show'] = False
    initial_registration_params['angle_step'] = 60
    initial_registration_params['cuda'] = False if params['device'] == "cpu" else True
    initial_registration_params['device'] = params['device']
    initial_registration_params['echo'] = False
    initial_registration_params['run_sift_ransac'] = True
    initial_registration_params['run_superpoint_superglue'] = True
    initial_registration_params['run_superpoint_ransac'] = False
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = False
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = False
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params


def default_initial_fast():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 2048
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "multi_feature"
    initial_registration_params['registration_size'] = 620
    initial_registration_params['registration_sizes'] = [150, 200, 250, 300, 400]
    initial_registration_params['transform_type'] = 'rigid'
    initial_registration_params['keypoint_threshold'] = 0.005
    initial_registration_params['match_threshold'] = 0.3
    initial_registration_params['sinkhorn_iterations'] = 50
    initial_registration_params['show'] = False
    initial_registration_params['angle_step'] = 180
    initial_registration_params['cuda'] = False if params['device'] == "cpu" else True
    initial_registration_params['device'] = params['device']
    initial_registration_params['echo'] = False
    initial_registration_params['run_sift_ransac'] = False
    initial_registration_params['run_superpoint_superglue'] = True
    initial_registration_params['run_superpoint_ransac'] = False
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = False
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = False
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params





















def default_nonrigid():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 4096
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "identity_initial_alignment"
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params
    
    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 4096
    nonrigid_registration_params['num_levels'] = 8
    nonrigid_registration_params['used_levels'] = 8
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100, 200]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params


def default_nonrigid_fast():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 2048
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "identity_initial_alignment"
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 2048
    nonrigid_registration_params['num_levels'] = 7
    nonrigid_registration_params['used_levels'] = 7
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params


def default_nonrigid_high_resolution():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 8192
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "identity_initial_alignment"
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 8192
    nonrigid_registration_params['num_levels'] = 9
    nonrigid_registration_params['used_levels'] = 9
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100, 200, 200]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015, 0.001]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8, 2.1] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params









































def default_initial_nonrigid():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 4096
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "multi_feature"
    initial_registration_params['registration_size'] = 620
    initial_registration_params['registration_sizes'] = [150, 200, 250, 300, 350, 400, 450, 500]
    initial_registration_params['transform_type'] = 'rigid'
    initial_registration_params['keypoint_threshold'] = 0.005
    initial_registration_params['match_threshold'] = 0.3
    initial_registration_params['sinkhorn_iterations'] = 50
    initial_registration_params['show'] = False
    initial_registration_params['angle_step'] = 90
    initial_registration_params['cuda'] = False if params['device'] == "cpu" else True
    initial_registration_params['device'] = params['device']
    initial_registration_params['echo'] = False
    initial_registration_params['run_sift_ransac'] = True
    initial_registration_params['run_superpoint_superglue'] = True
    initial_registration_params['run_superpoint_ransac'] = False
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 4096
    nonrigid_registration_params['num_levels'] = 8
    nonrigid_registration_params['used_levels'] = 8
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100, 200]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params


def default_initial_nonrigid_fast():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 2048
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "multi_feature"
    initial_registration_params['registration_size'] = 620
    initial_registration_params['registration_sizes'] = [150, 200, 250, 300, 350, 400]
    initial_registration_params['transform_type'] = 'rigid'
    initial_registration_params['keypoint_threshold'] = 0.005
    initial_registration_params['match_threshold'] = 0.3
    initial_registration_params['sinkhorn_iterations'] = 50
    initial_registration_params['show'] = False
    initial_registration_params['angle_step'] = 180
    initial_registration_params['cuda'] = False if params['device'] == "cpu" else True
    initial_registration_params['device'] = params['device']
    initial_registration_params['echo'] = False
    initial_registration_params['run_sift_ransac'] = False
    initial_registration_params['run_superpoint_superglue'] = True
    initial_registration_params['run_superpoint_ransac'] = False
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 2048
    nonrigid_registration_params['num_levels'] = 7
    nonrigid_registration_params['used_levels'] = 7
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params


def default_initial_nonrigid_high_resolution():
    params = dict()
    params['device'] = "cuda:0"
    params['echo'] = False
    params['save_final_images'] = True
    params['save_final_displacement_field'] = True
    params['case_name'] = "Default_Nonrigid"
    params['logging_path'] = None

    ### Loading Params ###
    loading_params = dict()
    loading_params['final_level'] = 0
    loading_params['pad_value'] = 255.0
    loading_params['loader'] = 'tiff'
    loading_params['source_resample_ratio'] = 0.2
    loading_params['target_resample_ratio'] = 0.2
    params['loading_params'] = loading_params

    ### Saving Params ###
    saving_params = dict()
    saving_params['saver'] = 'pil'
    saving_params['final_saver'] = 'tiff'
    saving_params['save_params'] = 'pil'
    saving_params['extension'] = ".jpg"
    saving_params['final_extension'] = ".tiff"
    params['saving_params'] = saving_params

    ### Preprocessing params ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = True
    preprocessing_params['initial_resolution'] = 8192
    preprocessing_params['normalization'] = True
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    preprocessing_params['save_results'] = True
    preprocessing_params['flip_intensity'] = True
    params['preprocessing_params'] = preprocessing_params

    ### Initial registration params ###
    run_initial_registration = True
    initial_registration_params = dict()
    initial_registration_params['save_results'] = True
    initial_registration_params['initial_registration_function'] = "multi_feature"
    initial_registration_params['registration_size'] = 620
    initial_registration_params['registration_sizes'] = [150, 200, 250, 300, 350, 400, 450, 500, 600]
    initial_registration_params['transform_type'] = 'rigid'
    initial_registration_params['keypoint_threshold'] = 0.005
    initial_registration_params['match_threshold'] = 0.3
    initial_registration_params['sinkhorn_iterations'] = 50
    initial_registration_params['show'] = False
    initial_registration_params['angle_step'] = 60
    initial_registration_params['cuda'] = False if params['device'] == "cpu" else True
    initial_registration_params['device'] = params['device']
    initial_registration_params['echo'] = False
    initial_registration_params['run_sift_ransac'] = True
    initial_registration_params['run_superpoint_superglue'] = True
    initial_registration_params['run_superpoint_ransac'] = False
    params['run_initial_registration'] = run_initial_registration
    params['initial_registration_params'] = initial_registration_params

    ### Nonrigid registration params ###
    run_nonrigid_registration = True
    nonrigid_registration_params = dict()
    nonrigid_registration_params['save_results'] = True
    nonrigid_registration_params['nonrigid_registration_function'] = "instance_optimization_nonrigid_registration"
    nonrigid_registration_params['device'] = params['device']
    nonrigid_registration_params['echo'] = False
    nonrigid_registration_params['cost_function'] = "ncc_local"
    nonrigid_registration_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_registration_params['regularization_function'] = "diffusion_relative"
    nonrigid_registration_params['regularization_function_params'] = dict()
    nonrigid_registration_params['registration_size'] = 8192
    nonrigid_registration_params['num_levels'] = 9
    nonrigid_registration_params['used_levels'] = 9
    nonrigid_registration_params['iterations'] = [100, 100, 100, 100, 100, 100, 100, 200, 200]
    nonrigid_registration_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015, 0.001]
    nonrigid_registration_params['alphas'] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8, 2.1] 
    params['run_nonrigid_registration'] = run_nonrigid_registration
    params['nonrigid_registration_params'] = nonrigid_registration_params
    ############################################
    return params








def save_params(params, parameters_path):
    with open(parameters_path, "w") as to_save:
        json.dump(params, to_save)

def load_parameters(parameters_path=None):
    if parameters_path is None:
        return create_identity() #TODO - replace with create_default()
    with open(parameters_path, "r") as to_load:
        params = json.loads(to_load.read())
    return params

if __name__ == "__main__":
    # save_params(default_initial(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_initial.json"))
    # save_params(default_initial_fast(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_initial_fast.json"))
    # save_params(default_nonrigid(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_nonrigid.json"))
    # save_params(default_nonrigid_fast(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_nonrigid_fast.json"))
    # save_params(default_nonrigid_high_resolution(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_nonrigid_high_resolution.json"))
    # save_params(default_initial_nonrigid(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_initial_nonrigid.json"))
    # save_params(default_initial_nonrigid_fast(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_initial_nonrigid_fast.json"))
    # save_params(default_initial_nonrigid_high_resolution(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_initial_nonrigid_high_resolution.json"))
    pass