#!/bin/bash
# Build and tag the research-graph-synthic image.

docker build . -t research-graph-synthetic -t gcr.io/research-graph/research-graph-synthetic
