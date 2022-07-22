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
from collections.abc import Iterable
import itertools
import random

import numpy as np
from sklearn.metrics import mean_squared_error

def MseWrapper(pred, true, scale=False):
  """Wrapper for Mean-Squared-Error eval metric.

  Arguments:
    pred: iterable of predicted values
    true: iterable of true values
    scale: if True, return value is divided by var(true)
  """
  mse = mean_squared_error(pred, true)
  if scale:
    var = np.var(true)
    if var != 0:
      mse /= var
  return mse


def ComputeNumPossibleConfigs(benchmark_params, h_params):
  num_possible_configs = 1
  if benchmark_params is not None:
    for _, value_list_or_value in benchmark_params.items():
      try:
        num_possible_configs *= len(value_list_or_value)
      except TypeError:
        continue
  if h_params is not None:
    for _, value_list_or_value in h_params.items():
      try:
        num_possible_configs *= len(value_list_or_value)
      except TypeError:
        continue
  return num_possible_configs


def _SampleValue(value_list_or_value):
  value = None
  if isinstance(value_list_or_value, str):
    value = value_list_or_value
  else:
    try:
      value = random.choice(value_list_or_value)
    except TypeError:
      value = value_list_or_value
  return value


def SampleModelConfig(benchmark_params, h_params):
  """Samples a model config from dictionaries of hyperparameter lists.

  """
  if benchmark_params is None:
    benchmark_params_sample = None
  else:
    benchmark_params_sample = {
      name: _SampleValue(value_list_or_value) for
      name, value_list_or_value in benchmark_params.items()
    }
  if h_params is None:
    h_params_sample = None
  else:
    h_params_sample = {
      name: _SampleValue(value_list_or_value) for
      name, value_list_or_value in h_params.items()
    }
  return benchmark_params_sample, h_params_sample

def GetCartesianProduct(param_list_dict):
  """Generator of full product space of the param lists."""
  sorted_names = list(sorted(param_list_dict))
  value_lists = [param_list_dict[k] for k in sorted_names]
  value_lists = [v if isinstance(v, Iterable) else [v] for v in value_lists]
  for element in itertools.product(*value_lists):
    yield dict(zip(sorted_names, element))


