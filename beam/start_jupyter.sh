#!/bin/bash

docker run -p 8888:8888 \
  -v ${PWD}/src:/app \
  research-graph-synthetic:latest \
  jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 8888 \
  --notebook-dir="/app/notebooks"
