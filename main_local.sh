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
BUILD_NAME="graphworld"
TASK="nodeclassification"
GENERATOR="sbm"
while getopts b:t:g: flag
do
    case "${flag}" in
        b) BUILD_NAME=${OPTARG};;
        t) TASK=${OPTARG};;
        g) GENERATOR=${OPTARG};;
    esac
done

OUTPUT_PATH="/tmp/${TASK}/${GENERATOR}"

rm -rf "${OUTPUT_PATH}"
mkdir -p ${OUTPUT_PATH}

ENTRYPOINT="python3 /app/beam_benchmark_main.py \
  --runner DirectRunner \
  --gin_files /app/configs/${TASK}_test.gin /app/configs/${TASK}_generators/${GENERATOR}/default_setup.gin \
  --output "${OUTPUT_PATH}""

echo ${ENTRYPOINT}

docker-compose run --entrypoint "${ENTRYPOINT}" ${BUILD_NAME}