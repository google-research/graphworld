#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Utilize docker-compose to run beam-pipeline locally in the same environment
# as the remote workers.
#
source launch_script_constants.sh
source local_job_setup.sh

TASK="nodeclassification"
GENERATOR="sbm"

while getopts t:g: flag
do
    case "${flag}" in
        t) TASK=${OPTARG};;
        g) GENERATOR=${OPTARG};;
    esac
done

OUTPUT_PATH="/tmp/${TASK}/${GENERATOR}"
rm -rf "${OUTPUT_PATH}"
mkdir -p ${OUTPUT_PATH}

# Add gin file string.
GIN_FILES="/app/configs/${TASK}.gin "
GIN_FILES="${GIN_FILES} /app/configs/${TASK}_generators/${GENERATOR}/default_setup.gin"
GIN_FILES="${GIN_FILES} /app/configs/common_hparams/${TASK}_test.gin"
if [ ${RUN_MODE2} = true ]; then
  GIN_FILES="${GIN_FILES} /app/configs/${TASK}_generators/${GENERATOR}/optimal_model_hparams.gin"
fi

# Add gin param string.
TASK_CLASS_NAME=$(get_task_class_name ${TASK})
GIN_PARAMS="GeneratorBeamHandlerWrapper.nsamples=${NUM_SAMPLES}\
            ${TASK_CLASS_NAME}BeamHandler.num_tuning_rounds=${NUM_TUNING_ROUNDS}\
            ${TASK_CLASS_NAME}BeamHandler.save_tuning_results=${SAVE_TUNING_RESULTS}"

ENTRYPOINT="python3 /app/beam_benchmark_main.py \
  --runner DirectRunner \
  --gin_files ${GIN_FILES} \
  --gin_params ${GIN_PARAMS} \
  --output "${OUTPUT_PATH}""

echo ${ENTRYPOINT}

docker-compose run --entrypoint "${ENTRYPOINT}" ${BUILD_NAME}