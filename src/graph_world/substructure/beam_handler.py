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
from sklearn.preprocessing import scale
from torch_geometric.data import DataLoader

from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..beam.generator_config_sampler import GeneratorConfigSampler, ParamSamplerSpec
from ..models.benchmarker import Benchmarker, BenchmarkGNNParDo
from .simulator import GenerateSubstructureDataset, GetSubstructureGraph, Substructure
from .utils import substructure_graph_to_torchgeo_data
from ..metrics.graph_metrics import graph_metrics

class SampleSubstructureDatasetDoFn(GeneratorConfigSampler, beam.DoFn):

  def __init__(self, param_sampler_specs, substruct, scale_labels=True,
               marginal=False):
    super(SampleSubstructureDatasetDoFn, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._AddSamplerFn('num_graphs', self._SampleUniformInteger)
    self._AddSamplerFn('num_vertices', self._SampleUniformInteger)
    self._AddSamplerFn('edge_prob', self._SampleUniformFloat)
    self._AddSamplerFn('train_prob', self._SampleUniformFloat)
    self._AddSamplerFn('tuning_prob', self._SampleUniformFloat)
    self._substruct = substruct
    self._scale_labels = scale_labels

  def process(self, sample_id):
    """Sample substructure dataset.
    """

    generator_config, marginal_param, fixed_params = self.SampleConfig(self._marginal)
    generator_config['generator_name'] = 'SubstructureDataset'

    data = GenerateSubstructureDataset(
      num_graphs=generator_config['num_graphs'],
      num_vertices=generator_config['num_vertices'],
      edge_prob=generator_config['edge_prob'],
      substruct_graph=GetSubstructureGraph(self._substruct)
    )

    if self._scale_labels:
      data['substruct_counts'] = scale(data['substruct_counts'])

    yield {'sample_id': sample_id,
           'marginal_param': marginal_param,
           'fixed_params': fixed_params,
           'generator_config': generator_config,
           'data': data}


class WriteSubstructureDoFn(beam.DoFn):

  def __init__(self, output_path):
    self._output_path = output_path

  # Let's not write all the graphs for now.
  def process(self, element):
    yield element


class ComputeSubstructureGraphMetricsParDo(beam.DoFn):

  def process(self, element):
    out = element
    graph_metrics_data = []
    graph_metrics_df = pd.DataFrame(
      data=[graph_metrics(graph) for graph in element['data']['graphs']])
    out['metrics'] = dict(graph_metrics_df.mean())
    yield out


class ConvertToTorchGeoDataParDo(beam.DoFn):
  def __init__(self, output_path, batch_size):
    self._output_path = output_path
    self._batch_size = batch_size

  def process(self, element):
    out = element
    torch_examples = []
    out['torch_dataset'] = {}
    out['skipped'] = False
    X = []
    y = []
    sample_id = element['sample_id']
    try:
      for graph, substruct_count in zip(
          element['data']['graphs'], element['data']['substruct_counts']):
        edge_density = 0.0
        if graph.num_edges() > 0:
          edge_density = graph.num_vertices() / graph.num_edges() ** 2.0
        X.append([edge_density])
        y.append(substruct_count)
        torch_examples.append(
          substructure_graph_to_torchgeo_data(graph, substruct_count))
    except:
      out['skipped'] = True
      print(f'failed to convert {sample_id}')
      logging.info(f'Failed to convert sbm_data to torchgeo for sample id {sample_id}')
      yield out
    X = np.array(X)
    y = np.array(y)
    num_train = int(len(torch_examples) * element['generator_config']['train_prob'])
    num_tuning = int(len(torch_examples) * element['generator_config']['tuning_prob'])
    out['torch_dataset'] = {
      'train': DataLoader(torch_examples[:num_train],
                          batch_size=self._batch_size, shuffle=True),
      'tuning': DataLoader(torch_examples[num_train:(num_train + num_tuning)],
                          batch_size=self._batch_size, shuffle=True),
      'test': DataLoader(torch_examples[(num_train + num_tuning):],
                         batch_size=self._batch_size, shuffle=False)
    }
    out['numpy_dataset'] = {
      'train': {'X': X[:num_train, :], 'y': y[:num_train]},
      'tuning': {'X': X[num_train:(num_train + num_tuning), :],
                 'y': y[num_train:(num_train + num_tuning)]},
      'test': {'X': X[(num_train + num_tuning):, :],
               'y': y[(num_train + num_tuning):]}
    }
    yield out


@gin.configurable
class SubstructureBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, substruct, benchmarker_wrappers, batch_size,
               marginal=False, num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False, scale_labels=False):
    self._sample_do_fn = SampleSubstructureDatasetDoFn(param_sampler_specs,
                                                       substruct, scale_labels, marginal)
    self._benchmark_par_do = BenchmarkGNNParDo(benchmarker_wrappers, num_tuning_rounds,
                                               tuning_metric, tuning_metric_is_loss)
    self._metrics_par_do = ComputeSubstructureGraphMetricsParDo()
    self._batch_size = batch_size

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
    self._write_do_fn = WriteSubstructureDoFn(output_path)
    self._convert_par_do = ConvertToTorchGeoDataParDo(output_path, self._batch_size)
    self._benchmark_par_do.SetOutputPath(output_path)