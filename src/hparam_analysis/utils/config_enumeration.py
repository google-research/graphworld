# Copyright 2020 Google LLC
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

from itertools import product
from typing import Optional, List

import gin

@gin.configurable
def enumerate_configs(
    hidden_channel_values: List[int] = [16],
    weight_decay_values: List[float] = [5e-5],
    dropout_values: List[float] = [0.5],
    learning_rate_values: List[float] = [1e-2],
    num_layers_values: List[float] = [3.0]
  ):

  configs = list(product(*[
    hidden_channel_values,
    weight_decay_values,
    dropout_values,
    learning_rate_values,
    num_layers_values
  ]))

  configs = [
    {'index': i,
     'hidden_channels': a,
     'weight_decay': b,
     'dropout': c,
     'learning_rate': d,
     'num_layers': g} for
    i, (a, b, c, d, g) in enumerate(configs)
  ]

  return configs


