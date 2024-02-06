import os
import pathlib


code_path = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


data_path = None # TODO
data_to_save = None # TODO
ANHIR_dataset_path = data_path / "ANHIR"
ACROBAT_dataset_path = data_path / "ACROBAT"
HyReCo_dataset_path = data_path / "HyReCo"

### Results ###
temporary_path = data_path / "Temporary"

acrobat_results_path = data_to_save / "ACROBAT"
anhir_results_path = data_to_save / "ANHIR"
hyreco_results_path = data_to_save / "HyReCo"

ACROBAT_training_data_path = ACROBAT_dataset_path / "Training"
ACROBAT_validation_data_path = ACROBAT_dataset_path / "Validation"
ACROBAT_testing_data_path = ACROBAT_dataset_path / "Testing"
ACROBAT_cbir_data_path = data_to_save / "ACROBAT" / "CBIR"
