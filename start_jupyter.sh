#!/bin/bash
# Copyright 2021 Google LLC
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


docker run -p 8888:8888 \
  -v ${PWD}/src:/app \
  -v /tmp:/tmp \
  --entrypoint /opt/venv/bin/jupyter \
  research-graph-synthetic:latest \
  notebook --allow-root --no-browser --port=8888 \
  --notebook-dir="/app/notebooks" --ip=0.0.0.0 
  
