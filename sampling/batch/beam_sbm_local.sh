#!/bin/bash

docker-compose run research-graph-synthetic python3 /app/beam_sbm.py \
  --output /app/data/beam_sbm \
  --runner=DirectRunner 
  # --job_endpoint=embed \
  # --environment_type=DOCKER \
  # --environment_config=gcr.io/research-graph/synthetic


