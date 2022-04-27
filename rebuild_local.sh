#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build and tag the build image.
# Same as ./build_local.sh just don't use the cached layers and start from scratch.

PROJECT_NAME="gcp-project-name"
BUILD_NAME="graphworld"

docker build --no-cache . -t ${BUILD_NAME}:latest -t gcr.io/${PROJECT_NAME}/${BUILD_NAME}:latest
