#!/bin/bash

OUTPUT_PATH="gs://research-graph-synthetic/${USER}/sampling/$(date +"%Y-%m-%d_%H-%M-%S")"
TEMP_LOCATION="${OUTPUT_PATH}/temp"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

NSAMPLES="${1:-10000}"
echo "NSAMPLES: ${NSAMPLES}"

ENTRYPOINT="python3 /app/beam_sbm.py \
  --runner=DataflowRunner \
  --project=research-graph \
  --region=us-east1 \
  --max_num_workers=256 \
  --temp_location="${TEMP_LOCATION}" \
  --output="${OUTPUT_PATH}" \
  --nsamples="${NSAMPLES}" \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/research-graph/research-graph-synthetic:latest"

echo "entrypoint: ${ENTRYPOINT}"

docker-compose run --entrypoint "${ENTRYPOINT}" research-graph-synthetic
  
  
