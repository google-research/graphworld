#!/bin/bash

docker run -p 8888:8888 \
  -v ${PWD}/src:/app \
  --entrypoint /opt/venv/bin/jupyter \
  research-graph-synthetic:latest \
  notebook --allow-root --no-browser --port=8888 \
  --notebook-dir="/app/notebooks" --ip=0.0.0.0 
  
