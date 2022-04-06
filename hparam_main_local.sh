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
BUILD_NAME="research-graph-synthetic"
OUTPUT_PATH="/tmp/hparam"
SIM=0

rm -rf "${OUTPUT_PATH}"
mkdir -p ${OUTPUT_PATH}

SIM_PREFIX=''
if [ ${SIM} = 0 ]
then
  SIM_PREFIX='no-';
fi;

docker-compose run \
  --entrypoint "python3 /app/hparam_analysis_main.py \
  --output ${OUTPUT_PATH} \
  --${SIM_PREFIX}sim \
  --gin_config=/app/configs/hparam_config.gin \
  --runner=DirectRunner" \
  ${BUILD_NAME}



