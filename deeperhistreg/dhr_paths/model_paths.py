import os
import pathlib


### Ready Models Paths ###
models_path =  pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dhr_models")))
superpoint_model_path = models_path / 'superpoint_v1.pth'
superglue_model_path = models_path / "superglue_outdoor.pth"
