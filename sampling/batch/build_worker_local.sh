#!/bin/bash
# Build and tag the research-graph-synthetic worker image.

docker build -f ./Dockerfile.BeamWorker . -t research-graph-synthetic-worker
docker tag research-graph-synthetic-worker gcr.io/research-graph/research-graph-synthetic-worker:latest
