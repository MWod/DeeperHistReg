"""
This example shows how to run the nonrigid registration using the library installed from PyPi (or manually from the repo). 
"""
import pathlib
from typing import Union

import deeperhistreg

def run():
    ### Define Inputs/Outputs ###
    source_path : Union[str, pathlib.Path] = None
    target_path : Union[str, pathlib.Path] = None
    output_path : Union[str, pathlib.Path] = None

    ### Define Params ###
    registration_params : dict = deeperhistreg.configs.default_nonrigid() # Alternative: # registration_params = deeperhistreg.configs.load_parameters(config_path) # To load config from JSON file
    save_displacement_field : bool = True # Whether to save the displacement field (e.g. for further landmarks/segmentation warping)
    copy_target : bool = True # Whether to copy the target (e.g. to simplify the further analysis
    delete_temporary_results : bool = True # Whether to keep the temporary results
    case_name : str = "Example_Nonrigid" # Used only if the temporary_path is important, otherwise - provide whatever
    temporary_path : Union[str, pathlib.Path] = None # Will use default if set to None

    ### Create Config ###
    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['output_path'] = output_path
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path
    
    ### Run Registration ###
    deeperhistreg.run_registration(**config)

if __name__ == "__main__":
    run()