#!/bin/bash

TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
OUTPUT_PATH="gs://research-graph-synthetic/${USER}/sampling/${TIMESTAMP}"
TEMP_LOCATION="${OUTPUT_PATH}/temp"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

NSAMPLES="${1:-10000}"
echo "NSAMPLES: ${NSAMPLES}"

JOB_NAME=$(echo "${USER}-${TIMESTAMP}" | tr '_' '-')

ENTRYPOINT="python3 /app/beam_sbm.py \
  --runner=DataflowRunner \
  --project=research-graph \
  --region=us-east1 \
  --max_num_workers=256 \
  --temp_location="${TEMP_LOCATION}" \
  --output="${OUTPUT_PATH}" \
  --nsamples="${NSAMPLES}" \
  --job_name="${JOB_NAME}" \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/research-graph/research-graph-synthetic:latest"

echo "entrypoint: ${ENTRYPOINT}"

docker-compose run --entrypoint "${ENTRYPOINT}" research-graph-synthetic
  
  
