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
from torch_geometric.data import DataLoader

from ..beam.benchmarker import BenchmarkGNNParDo
from ..beam.generator_beam_handler import GeneratorBeamHandler
from .utils import graph_regression_dataset_example_to_torch_geo_data
from ..metrics.graph_metrics import graph_metrics


class SampleGraphRegressionDatasetDoFn(beam.DoFn):

  def __init__(self, generator_wrapper):
    self._generator_wrapper = generator_wrapper

  def process(self, sample_id):
    """Sample generator outputs."""
    yield self._generator_wrapper.Generate(sample_id)


class WriteGraphRegressionDatasetDoFn(beam.DoFn):

  def __init__(self, output_path):
    self._output_path = output_path

  # Let's not write all the graphs for now.
  def process(self, element):
    yield element


class ComputeGraphRegressionMetricsParDo(beam.DoFn):

  def process(self, element):
    out = element
    graph_metrics_df = pd.DataFrame(
      data=[graph_metrics(graph) for graph in element['data'].graphs])
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
    # try:
    for graph, features, target in zip(
        element['data'].graphs,
        element['data'].graph_node_features,
        element['data'].graph_regression_target):
      edge_density = 0.0
      if graph.num_edges() > 0:
        edge_density = graph.num_vertices() / graph.num_edges() ** 2.0
      X.append([edge_density])
      y.append(target)
      torch_examples.append(
        graph_regression_dataset_example_to_torch_geo_data(
            graph, target, features))
    # except:
    #   out['skipped'] = True
    #   print(f'failed to convert {sample_id}')
    #   logging.info(
    #       f'Failed to convert sbm_data to torchgeo for sample id {sample_id}')
    #   yield out
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
class GraphRegressionBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, benchmarker_wrappers, generator_wrapper, batch_size,
               num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False):
    self._sample_do_fn = SampleGraphRegressionDatasetDoFn(generator_wrapper)
    self._benchmark_par_do = BenchmarkGNNParDo(
        benchmarker_wrappers, num_tuning_rounds, tuning_metric,
        tuning_metric_is_loss)
    self._metrics_par_do = ComputeGraphRegressionMetricsParDo()
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
    self._write_do_fn = WriteGraphRegressionDatasetDoFn(output_path)
    self._convert_par_do = ConvertToTorchGeoDataParDo(output_path,
                                                      self._batch_size)
    self._benchmark_par_do.SetOutputPath(output_path)