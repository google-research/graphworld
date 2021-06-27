#!/bin/bash
# Kick off a build on GCP.
#
gcloud builds submit --tag gcr.io/research-graph/research-graph-synthetic
