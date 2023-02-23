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

PROJECT_NAME="project"
BUILD_NAME="graphworld"
SIM=1
MACHINE_TYPE="n1-standard-1"
MAX_NUM_WORKERS=1000
DATASET_NAME="cora"
while getopts p:b:m:w: flag
do
    case "${flag}" in
        p) PROJECT_NAME=${OPTARG};;
        b) BUILD_NAME=${OPTARG};;
        m) MACHINE_TYPE=${OPTARG};;
        w) MAX_NUM_WORKERS=${OPTARG};;
    esac
done

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
JOB_NAME="hparam-${DATASET_NAME}-${TIMESTAMP}"
OUTPUT_PATH="gs://${BUILD_NAME}/${USER}/sampling/${JOB_NAME}"
DATASET_PATH="gs://${BUILD_NAME}/npz-datasets"
TEMP_LOCATION="gs://${BUILD_NAME}/temp"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

FULL_JOB_NAME=$(echo "${USER}-${JOB_NAME}" | tr '_' '-')

SIM_PREFIX=''
if [ ${SIM} = 0 ]
then
  SIM_PREFIX='no-';
fi;

ENTRYPOINT="python3 /app/hparam_analysis_main.py \
  --runner=DataflowRunner \
  --project=${PROJECT_NAME} \
  --region=us-east1 \
  --max_num_workers="${MAX_NUM_WORKERS}" \
  --temp_location="${TEMP_LOCATION}" \
  --gin_config=/app/configs/hparam_config.gin \
  --dataset_path="${DATASET_PATH}" \
  --dataset_name="${DATASET_NAME}" \
  --output="${OUTPUT_PATH}" \
  --${SIM_PREFIX}sim \
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
