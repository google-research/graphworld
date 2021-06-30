#!/bin/bash

TEMP_LOCATION="gs://research-graph-synthetic/temp"


OUTPUT_PATH="gs://research-graph-synthetic/${USER}/sampling/$(date +"%Y-%m-%d_%H-%M-%S")"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

python3 ./src/beam_sbm.py \
  --runner=DataflowRunner \
  --project=research-graph \
  --region=us-east1 \
  --temp_location="${TEMP_LOCATION}" \
  --output="${OUTPUT_PATH}" \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/research-graph/research-graph-synthetic:latest
