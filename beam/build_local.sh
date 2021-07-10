#!/bin/bash
# Build and tag the research-graph-synthic image.

docker build . -t research-graph-synthetic:latest -t gcr.io/research-graph/research-graph-synthetic:latest
