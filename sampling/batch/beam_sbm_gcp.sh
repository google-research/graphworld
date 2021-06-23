#!/bin/bash

  # --setup_file=./beam_setup.py \
python3 ./src/beam_sbm.py \
  --runner=DataflowRunner \
  --project=research-graph \
  --region=us-east1 \
  --temp_location=gs://research-graph-synthetic/temp \
  --output=gs://research-graph-synthetic/exp2 \
  --experiments=use_monitoring_state_manager \
  --experiments=enable_execution_details_collection \
  --experiment=use_runner_v2 \
  --worker_harness_container_image=gcr.io/research-graph/research-graph-synthetic:latest
  
  
