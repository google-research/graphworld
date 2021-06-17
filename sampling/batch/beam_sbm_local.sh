#!/bin/bash

docker-compose run research-graph-synthetic python3 /app/beam_sbm.py --input_collection /app/data/random_configs/configs_collection.txt --runner PortableRunner
