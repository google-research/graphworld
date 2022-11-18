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

from ..beam.benchmarker import Benchmarker, BenchmarkGNNParDo
from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..metrics.graph_metrics import graph_metrics
from ..metrics.node_label_metrics import NodeLabelMetrics
from ..linkprediction.utils import linkprediction_data_to_torchgeo_data


class SampleLinkPredictionDatasetDoFn(beam.DoFn):

  def __init__(self, generator_wrapper):
    self._generator_wrapper = generator_wrapper

  def process(self, sample_id):
    """Sample generator outputs."""
    yield self._generator_wrapper.Generate(sample_id)


class WriteLinkPredictionDatasetDoFn(beam.DoFn):

  def __init__(self, output_path):
    self._output_path = output_path

  def process(self, element):
    sample_id = element['sample_id']
    config = element['generator_config']
    data = element['data']

    text_mime = 'text/plain'
    prefix = '{0:05}'.format(sample_id)
    config_object_name = os.path.join(self._output_path, prefix + '_config.txt')
    with beam.io.filesystems.FileSystems.create(
        config_object_name, text_mime) as f:
      buf = bytes(json.dumps(config), 'utf-8')
      f.write(buf)
      f.close()

    graph_object_name = os.path.join(self._output_path, prefix + '_graph.gt')
    with beam.io.filesystems.FileSystems.create(graph_object_name) as f:
      data.graph.save(f)
      f.close()

    graph_memberships_object_name = os.path.join(
      self._output_path, prefix + '_graph_memberships.txt')
    with beam.io.filesystems.FileSystems.create(
        graph_memberships_object_name, text_mime) as f:
      np.savetxt(f, data.graph_memberships)
      f.close()

    node_features_object_name = os.path.join(
      self._output_path, prefix + '_node_features.txt')
    with beam.io.filesystems.FileSystems.create(
        node_features_object_name, text_mime) as f:
      np.savetxt(f, data.node_features)
      f.close()

    edge_features_object_name = os.path.join(
      self._output_path, prefix + '_edge_features.txt')
    with beam.io.filesystems.FileSystems.create(
        edge_features_object_name, text_mime) as f:
      for edge_tuple, features in data.edge_features.items():
        buf = bytes('{0},{1},{2}'.format(
            edge_tuple[0], edge_tuple[1], features), 'utf-8')
        f.write(buf)
      f.close()


class ComputeLinkPredictionMetrics(beam.DoFn):

  def process(self, element):
    out = element
    out['metrics'] = graph_metrics(element['data'].graph)
    out['metrics'].update(NodeLabelMetrics(element['data'].graph,
                                            element['data'].graph_memberships,
                                            element['data'].node_features))
    yield out


class ConvertToTorchGeoDataParDo(beam.DoFn):

  def __init__(self, training_ratio, tuning_ratio):
    self._training_ratio = training_ratio
    self._tuning_ratio = tuning_ratio

  def process(self, element):
    sample_id = element['sample_id']
    linkprediction_data = element['data']


    out = {
        'sample_id': sample_id,
        'metrics': element['metrics'],
        'torch_data': None,
        'masks': None,
        'skipped': False,
        'generator_config': element['generator_config'],
        'marginal_param': element['marginal_param'],
        'fixed_params': element['fixed_params']
    }

    try:
      torch_data = linkprediction_data_to_torchgeo_data(
          linkprediction_data, self._training_ratio, self._tuning_ratio)
      out['torch_data'] = torch_data
    except:
      out['skipped'] = True
      print(f'failed to convert {sample_id}')
      logging.info(
           ('Failed to convert linkprediction_data to torchgeo'
            'for sample id %d'), sample_id)
      yield out
      return

    yield out


@gin.configurable
class LinkPredictionBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, benchmarker_wrappers, generator_wrapper,
               training_ratio, tuning_ratio,
               marginal=False, num_tuning_rounds=1, tuning_metric='',
               tuning_metric_is_loss=False, save_tuning_results=False):
    self._sample_do_fn = SampleLinkPredictionDatasetDoFn(generator_wrapper)
    self._benchmark_par_do = BenchmarkGNNParDo(
        benchmarker_wrappers, num_tuning_rounds, tuning_metric,
        tuning_metric_is_loss, save_tuning_results)
    self._metrics_par_do = ComputeLinkPredictionMetrics()
    self._training_ratio = training_ratio
    self._tuning_ratio = tuning_ratio

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
    self._write_do_fn = WriteLinkPredictionDatasetDoFn(output_path)
    self._convert_par_do = ConvertToTorchGeoDataParDo(self._training_ratio,
                                                      self._tuning_ratio)
    self._benchmark_par_do.SetOutputPath(output_path)
