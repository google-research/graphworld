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
source launch_script_constants.sh
source remote_job_setup.sh

TASK="nodeclassification"
GENERATOR="sbm"
JOBTAG="taghere"

while getopts t:g:j: flag
do
    case "${flag}" in
        t) TASK=${OPTARG};;
        g) GENERATOR=${OPTARG};;
        j) JOBTAG=${OPTARG};;
    esac
done

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
JOB_NAME="${TASK}-${GENERATOR}-${TIMESTAMP}-${JOBTAG}"
OUTPUT_PATH="gs://${BUILD_NAME}/${USER}/sampling/${JOB_NAME}"
TEMP_LOCATION="gs://${BUILD_NAME}/temp"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
FULL_JOB_NAME=$(echo "${USER}-${JOB_NAME}" | tr '_' '-')

# Add gin file string.
GIN_FILES="/app/configs/${TASK}.gin "
GIN_FILES="${GIN_FILES} /app/configs/${TASK}_generators/${GENERATOR}/default_setup.gin"
GIN_FILES="${GIN_FILES} /app/configs/common_hparams/${TASK}.gin"
if [ ${RUN_MODE2} = true ]; then
  GIN_FILES="${GIN_FILES} /app/configs/${TASK}_generators/${GENERATOR}/optimal_model_hparams.gin"
fi

# Add gin param string.
TASK_CLASS_NAME=$(get_task_class_name ${TASK})
GIN_PARAMS="GeneratorBeamHandlerWrapper.nsamples=${NUM_SAMPLES}\
            ${TASK_CLASS_NAME}BeamHandler.num_tuning_rounds=${NUM_TUNING_ROUNDS}\
            ${TASK_CLASS_NAME}BeamHandler.save_tuning_results=${SAVE_TUNING_RESULTS}"

ENTRYPOINT="python3 /app/beam_benchmark_main.py \
  --runner=DataflowRunner \
  --project=${PROJECT_NAME} \
  --region=us-east1 \
  --max_num_workers="${MAX_NUM_WORKERS}" \
  --temp_location="${TEMP_LOCATION}" \
  --gin_files "${GIN_FILES}" \
  --gin_params "${GIN_PARAMS}" \
  --output="${OUTPUT_PATH}" \
  --job_name="${FULL_JOB_NAME}" \
  --no_use_public_ips \
  --network=dataflow-vpc \
  --worker_machine_type="${MACHINE_TYPE}" \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/${PROJECT_NAME}/${BUILD_NAME}:latest"

echo "entrypoint: ${ENTRYPOINT}"

docker-compose run --detach --entrypoint "${ENTRYPOINT}" ${BUILD_NAME}
