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

import logging

import apache_beam as beam
import gin
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..beam.generator_config_sampler import GeneratorConfigSampler, \
  ParamSamplerSpec
from ..models.benchmarker import Benchmarker, BenchmarkGNNParDo
from ..sbm.beam_handler import SampleSbmDoFn, WriteSbmDoFn, \
  ComputeSbmGraphMetrics
from ..sbm.utils import sbm_data_to_torchgeo_data
from .utils import calculate_target, sample_masks


class ConvertToTorchGeoDataParDo(beam.DoFn):
  def __init__(self, target, training_ratio, tuning_ratio, normalize_target):
    self._target = target
    self._training_ratio = training_ratio
    self._tuning_ratio = tuning_ratio
    self._normalize_target = normalize_target

  def process(self, element):
    sample_id = element['sample_id']
    sbm_data = element['data']
    if isinstance(self._target, list):
      self._target = random.choice(self._target)
    out = {
        'sample_id': sample_id,
        'metrics': element['metrics'],
        'torch_data': None,
        'masks': None,
        'target': self._target,
        'skipped': False,
        'generator_config': element['generator_config'],
        'marginal_param': element['marginal_param'],
        'fixed_params': element['fixed_params']
    }

    try:
      torch_data = sbm_data_to_torchgeo_data(sbm_data)
      y = calculate_target(sbm_data.graph, self._target)
      if self._normalize_target:
        y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
      torch_data.y = torch.tensor(y, dtype=torch.float)
      out['torch_data'] = torch_data
      out['masks'] = sample_masks(y.shape[0], self._training_ratio,
                                  self._tuning_ratio)
    except Exception as e:
      out['skipped'] = True
      print(f'failed to convert {sample_id}', e)
      logging.info(
          f'Failed to convert sbm_data to torchgeo for sample id {sample_id}',
          e)
      yield out
      return

    yield out


@gin.configurable
class NodeRegressionBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, benchmarker_wrappers, target,
      training_ratio, tuning_ratio, marginal=False,
      num_tuning_rounds=1, tuning_metric='', normalize_target=True,
      tuning_metric_is_loss=False, save_tuning_results=False):
    self._sample_do_fn = SampleSbmDoFn(param_sampler_specs, marginal)
    self._benchmark_par_do = BenchmarkGNNParDo(benchmarker_wrappers,
                                               num_tuning_rounds, tuning_metric,
                                               tuning_metric_is_loss,
                                               save_tuning_results)
    self._target = target
    self._metrics_par_do = ComputeSbmGraphMetrics()
    self._training_ratio = training_ratio
    self._tuning_ratio = tuning_ratio
    self._save_tuning_results = save_tuning_results
    self._normalize_target = normalize_target

  def GetSampleDoFn(self):
    return self._sample_do_fn

  def GetWriteDoFn(self):
    return self._write_do_fn

  def GetConvertParDo(self):
    return self._convert_par_do

  def GetBenchmarkParDo(self):
    return self._benchmark_par_do

  def GetGraphMetricsParDo(self):
    return self._metrics_par_do

  def SetOutputPath(self, output_path):
    self._output_path = output_path
    self._write_do_fn = WriteSbmDoFn(output_path)
    self._convert_par_do = ConvertToTorchGeoDataParDo(self._target,
                                                      self._training_ratio,
                                                      self._tuning_ratio,
                                                      self._normalize_target)
    self._benchmark_par_do.SetOutputPath(output_path)
