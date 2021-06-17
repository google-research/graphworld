#!/bin/bash

docker-compose run research-graph-synthetic python3 /app/generate_random_configs.py --gin_bindings SampleSmbConfig.nsamples=20
