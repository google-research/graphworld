#!/bin/bash

docker-compose run --entrypoint "python3 /app/generate_stochastic_block_model_with_features.py --gin_files /app/configs/smb_test.gin" research-graph-synthetic
