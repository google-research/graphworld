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
    return np.random.randint(int(param_sampler.min_val), int(param_sampler.max_val))

  def _SampleUniformFloat(self, param_sampler):
    return np.random.uniform(param_sampler.min_val, param_sampler.max_val)

  def _AddSamplerFn(self, param_name, sampler_fn):
    if param_name not in self._param_sampler_specs:
      raise InvalidArgumentError("param %s not found in input param specs" % param_name)
    self._param_sampler_specs[param_name].sampler_fn = sampler_fn

  def __init__(self, param_sampler_specs):
    self._param_sampler_specs = {spec.name: spec for spec in param_sampler_specs}

  def SampleConfig(self):
    config = {}
    for param_name, spec in self._param_sampler_specs.items():
      config[param_name] = spec.sampler_fn(spec)
    return config