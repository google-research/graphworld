#!/bin/bash

docker-compose run research-graph-synthetic python3 /app/generate_stochastic_block_model_with_features.py --gin_files /app/configs/smb_test.gin
