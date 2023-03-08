#!/bin/bash

# This file defines variables needed for main_gcp.sh that are commonly
# shared across GraphWorld tasks and generators.

PROJECT_NAME="project"
BUILD_NAME="graphworld"
MACHINE_TYPE="n1-standard-1"
MAX_NUM_WORKERS=1000
# bash boolean
RUN_MODE2=false
NUM_SAMPLES=10
NUM_TUNING_ROUNDS=1
# py boolean
SAVE_TUNING_RESULTS=False