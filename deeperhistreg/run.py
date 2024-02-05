### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from typing import Iterable
import logging
import argparse
from datetime import datetime
import shutil
import pathlib

### External Imports ###

### Internal Imports ###
from dhr_pipeline import full_resolution as fr
from dhr_pipeline import registration_params as rp


def run_registration(**config):
    ### Parse Config ###
    try:
        registration_parameters_path = config['registration_parameters_path']
        registration_parameters = rp.load_parameters(registration_parameters_path)
    except KeyError:
        registration_parameters = config['registration_parameters']

    source_path = config['source_path']
    target_path = config['target_path']
    output_path = config['output_path']
    experiment_name = config['case_name']
    save_path = config['temporary_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(datetime.now()))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### Run Registration ###
    try:
        registration_parameters['logging_path'] = pathlib.Path(save_path) / "logs.txt"
        registration_parameters['case_name'] = experiment_name
        pipeline = fr.DeeperHistReg_FullResolution(registration_parameters)
        pipeline.run_registration(source_path, target_path, save_path)
    except Exception as e:
        print(f"Exception: {e}")

    ### Copy Outputs and Clean ###
    if registration_parameters['save_final_images']:
        warped_name = [item for item in os.listdir(pathlib.Path(save_path) / experiment_name / "Results_Final") if "warped_source" in item][0]
        shutil.copy(pathlib.Path(save_path) / experiment_name / "Results_Final" / warped_name, pathlib.Path(output_path) / warped_name)
        shutil.copy(pathlib.Path(save_path) / "logs.txt", pathlib.Path(output_path) / "logs.txt")
        if config['copy_target']:
            _, target_name = os.path.split(target_path)
            _, extension = os.path.splitext(target_name)
            target_name = "target" + extension
            shutil.copy(target_path, pathlib.Path(output_path) / target_name)

    if config['save_displacement_field']:
        shutil.copy(pathlib.Path(save_path) / experiment_name / "Results_Final" / "displacement_field.mha", pathlib.Path(output_path) / "displacement_field.mha")

    try:
        shutil.copy(pathlib.Path(save_path) / experiment_name / "Results_Final" / "postprocessing_params.json", pathlib.Path(output_path) / "postprocessing_params.json")
    except:
        pass

    if config['delete_temporary_results']:
        shutil.rmtree(save_path)

def parse_args(args : Iterable) -> dict:
    ### Create Parser ###
    parser = argparse.ArgumentParser(description="DeeperHistReg arguments")

    ### Core ###
    parser.add_argument('--srcp', dest='source_path', type=str, help="Path to the source image")
    parser.add_argument('--trgp', dest='target_path', type=str, help="Path to the target image")
    parser.add_argument('--out', dest='output_path', type=str, help="Path to the output folder")

    ### Optional ###
    nonrigid_parameters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deeperhistreg_params", "default_nonrigid.json")
    parser.add_argument('--params', dest='registration_parameters_path', type=str, default=nonrigid_parameters_path, help="Path to the JSON with registration parameters")
    parser.add_argument('--exp', dest='case_name', type=str, default="WSI_Registration", help="Case name")
    parser.add_argument('--sdf', dest='save_displacement_field', action='store_true', help="Save displacement field?")
    parser.add_argument('--cpt', dest='copy_target', action='store_true', help="Copy target image?")
    parser.add_argument('--dtmp', dest='delete_temporary_results', action='store_true', help='Delete temporary results?')
    parser.add_argument('--temp', dest='temporary_path', type=str, default=None,
                        help='Path to save the temporary results. Defaults to random folder inside the file directory.')

    ### Parse Parameters ###
    config = parser.parse_args()
    config = vars(config)
    return config

if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    run_registration(**config)
