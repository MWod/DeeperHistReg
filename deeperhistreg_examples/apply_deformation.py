"""
This example shows how to run the nonrigid registration and then apply the calculated displacement field to chosen image (e.g. segmentation mask)
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
    registration_params['save_final_images'] = False # The warped images will not be saved during the registration

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

    ### Apply Warping ###
    source_image_path = source_path # Here the same as during the registration - can be replaced with path to the binary mask
    target_image_path = target_path # Here the same as during the registration - serves as a template
    warped_image_path = None # Where to save the warped image
    displacement_field_path = pathlib.Path(output_path) / "displacement_field.mha"
    loader = deeperhistreg.loaders.OpenSlideLoader
    saver = deeperhistreg.savers.TIFFSaver
    save_params = deeperhistreg.savers.tiff_saver.default_params
    level = 0 # Pyramid level to perform the warping - 0 is the highest possible
    pad_value = 255 # White in the RGB representation
    save_source_only = True # Whether to save only the warped image or also the corresponding target image
    to_template_shape = True # Whether to align the source shape to template shape (if initially different)
    to_save_target_path = None # Path where to save the target (if save_source_only set to False)

    deeperhistreg.apply_deformation(
        source_image_path = source_image_path,
        target_image_path = target_image_path,
        warped_image_path = warped_image_path, 
        displacement_field_path = displacement_field_path,
        loader = loader,
        saver = saver,
        save_params = save_params,
        level = level,
        pad_value = pad_value,
        save_source_only = save_source_only,
        to_template_shape = to_template_shape,
        to_save_target_path = to_save_target_path
    )





if __name__ == "__main__":
    run()