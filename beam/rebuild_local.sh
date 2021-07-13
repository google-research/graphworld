#!/bin/bash
# Build and tag the research-graph-synthic image.
# Same as ./build_local.sh just don't use the cached layers and start from scratch.

docker build --no-cache . -t research-graph-synthetic:latest -t gcr.io/research-graph/research-graph-synthetic:latest
