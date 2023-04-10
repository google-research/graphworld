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

import random

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import gin
import numpy as np
from typing import Any


@gin.configurable
@dataclass
class ParamSamplerSpec:
  name: str
  sampler_fn: Any = None
  min_val: float = -np.inf
  max_val: float = np.inf
  default_val: float = None


class GeneratorConfigSampler:
  # Base class for sampling generator configs.
  #
  # A child class C should call the following in its __init__:
  #   super(C, self).__init__(param_sample_specs)
  # Following this line, add sampler_fns to each spec with:
  #   self._AddSampleFn('foo', self._SampleUniformInteger)
  #   self._AddSampleFn('bar', self._SampleUniformFloat)
  # The sampler_fn can also be any function accessible to the child class.
  #
  # Arguments:
  #   param_sampler_specs: a list of ParamSamplerSpecs.

  def _SampleUniformInteger(self, param_sampler):
    low = int(param_sampler.min_val)
    high = int(param_sampler.max_val)
    if high < low:
      raise RuntimeError(
        "integer sampling for %s failed as high < low" % param_sampler.name)
    return low if low == high else np.random.randint(low, high)

  def _SampleUniformFloat(self, param_sampler):
    return np.random.uniform(param_sampler.min_val, param_sampler.max_val)

  def _AddSamplerFn(self, param_name, sampler_fn):
    if param_name not in self._param_sampler_specs:
      raise RuntimeError("param %s not found in input param specs" % param_name)
    self._param_sampler_specs[param_name].sampler_fn = sampler_fn

  def _ChooseMarginalParam(self):
    valid_params = [
      param_name for
      param_name, spec in self._param_sampler_specs.items() if
      spec.min_val != spec.max_val]
    if len(valid_params) == 0:
      return None
    return random.choice(valid_params)

  def __init__(self, param_sampler_specs):
    self._param_sampler_specs = {spec.name: spec for spec in param_sampler_specs}

  def SampleConfig(self, marginal=False):
    config = {}
    marginal_param = None
    if marginal:
      marginal_param = self._ChooseMarginalParam()
    fixed_params = []
    for param_name, spec in self._param_sampler_specs.items():
      param_value = None
      if marginal and marginal_param is not None:
        # If the param is not a marginal param, give it its default (if possible)
        if param_name != marginal_param:
          if spec.default_val is not None:
            fixed_params.append(param_name)
            param_value = spec.default_val
      # If the param val is still None, give it a random value.
      if param_value is None:
        # Allow parameters not being sampled to remain in output as null
        if spec.sampler_fn is None: 
          param_value = None
        else: 
          param_value = spec.sampler_fn(spec)
      config[param_name] = param_value
    return config, marginal_param, fixed_params

  def _AddParamSamplerSpecs(self, extra_param_sampler_specs, overwrite=False):
    if extra_param_sampler_specs is not None:
      # Merging dictionaries values whilst overwritting shared values if needed
      if overwrite:
        self._param_sampler_specs = {**self._param_sampler_specs, **{spec.name: spec for spec in extra_param_sampler_specs}}
      else:
        self._param_sampler_specs = {**{spec.name: spec for spec in extra_param_sampler_specs}, **self._param_sampler_specs}