#!/bin/bash
: '
This example shows how to use the the DeeperHistReg library using the Docker container (single registration).
You can use the provided image or build it by yourself.
To build the docker directly use: docker build --tag image_name . <- where image_name is the desired name.
'
IMAGE_NAME=deeperhistreg # The container name

SOURCE_PATH=/source_path  # Full path to the source image
TARGET_PATH=/target_path # Full path to the target image
PARAMS=/path_to_json_params.json # Full path to the registration config (see deeperhistreg_params) - Defaults to default_nonrigid.json
SAVE_PATH=/save_directory # Full path to the directory that will contain the results (will be created if does not exists)
CASE_NAME=Example # Name for the given registration pair
TEMP_DIR=/temporary_directory # Temporary results - contain intermediate registration steps (defaults to reasonable value)
# --sdf - Whether to save the displacement field (e.g. for further landmarks warping or segmentation transfer)
# --cpt - Whether to copy the target to the output folder (to simplify further analysis)
# --dtmp - Whether to delete the temporary results (presenting preprocessing, initial alignement, nonrigid registration - makes sense for Docker)

docker run --gpus all -v $SOURCE_PATH:$SOURCE_PATH -v $TARGET_PATH:$TARGET_PATH -v $PARAMS:$PARAMS -v $SAVE_PATH:$SAVE_PATH $IMAGE_NAME \
 --srcp $SOURCE_PATH --trgp $TARGET_PATH --out $SAVE_PATH --params $PARAMS --exp $CASE_NAME --temp $TEMP_DIR --sdf --cpt --dtmp