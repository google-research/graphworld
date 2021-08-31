
import argparse
import json
import logging
import os

import apache_beam as beam
import gin
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader

from generator_beam_handler import GeneratorBeamHandler
from generator_config_sampler import GeneratorConfigSampler, ParamSamplerSpec
from models.wrappers import LinearGraphGCN
from substructure.simulator import GenerateSubstructureDataset, GetSubstructureGraph, Substructure
from substructure.utils import substructure_graph_to_torchgeo_data
from graph_metrics import GraphMetrics

class SampleSubstructureDatasetDoFn(GeneratorConfigSampler, beam.DoFn):

  def __init__(self, param_sampler_specs, substruct):
    super(SampleSubstructureDatasetDoFn, self).__init__(param_sampler_specs)
    self._AddSamplerFn('num_graphs', self._SampleUniformInteger)
    self._AddSamplerFn('num_vertices', self._SampleUniformInteger)
    self._AddSamplerFn('edge_prob', self._SampleUniformFloat)
    self._AddSamplerFn('train_prob', self._SampleUniformFloat)
    self._substruct = substruct

  def process(self, sample_id):
    """Sample substructure dataset.
    """

    generator_config = self.SampleConfig()
    generator_config['generator_name'] = 'SubstructureDataset'

    data = GenerateSubstructureDataset(
      num_graphs=generator_config['num_graphs'],
      num_vertices=generator_config['num_vertices'],
      edge_prob=generator_config['edge_prob'],
      substruct_graph=GetSubstructureGraph(self._substruct)
    )

    yield {'sample_id': sample_id,
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
      data=[GraphMetrics(graph) for graph in element['data']['graphs']])
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
    try:
      for graph, substruct_count in zip(
          element['data']['graphs'], element['data']['substruct_counts']):
        torch_examples.append(
          substructure_graph_to_torchgeo_data(graph, substruct_count))
    except:
      out['skipped'] = True
      logging.info(f'Failed to sample masks for sample id {sample_id}')
      yield out
    num_train = int(len(torch_examples) * element['generator_config']['train_prob'])
    out['torch_dataset'] = {
      'train': DataLoader(torch_examples[:num_train],
                          batch_size=self._batch_size, shuffle=True),
      'test': DataLoader(torch_examples[num_train:],
                         batch_size=self._batch_size, shuffle=False)
    }
    yield out


class BenchmarkGNNParDo(beam.DoFn):
  def __init__(self, num_features, hidden_channels=64, epochs=100, lr=0.0001):
    self._num_features = num_features
    self._hidden_channels = hidden_channels
    self._epochs = epochs
    self._lr = lr

  def SetOutputPath(self, output_path):
    self._output_path = output_path

  def process(self, element):

    linear_model = LinearGraphGCN(
      self._num_features,
      self._hidden_channels,
      self._epochs,
      self._lr
    )

    print("type of element['torch_dataset']['train'] is %s" % type(element['torch_dataset']['train']))
    print("element['torch_dataset']['train'] is %s" % str(element['torch_dataset']['train']))
    mses, losses = linear_model.train(element['torch_dataset']['train'])
    test_mse = None
    try:
      test_mse = float(linear_model.test(element['torch_dataset']['test']))
    except:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')

    sample_id = element['sample_id']
    benchmark_result = {
      'sample_id': sample_id,
      'losses': losses,
      'test_mse': test_mse,
      'generator_config': element['generator_config']
    }
    for name, result in benchmark_result.items():
      print('benchmark result %s has type %s' % (name, type(result)))

    results_object_name = os.path.join(self._output_path, '{0:05}_results.txt'.format(sample_id))
    with beam.io.filesystems.FileSystems.create(results_object_name, 'text/plain') as f:
      buf = bytes(json.dumps(benchmark_result), 'utf-8')
      f.write(buf)
      f.close()

    test_mse = (0.0 if benchmark_result['test_mse'] is None else
                benchmark_result['test_mse'])
    output_data = {"test_mse": test_mse}
    output_data.update(benchmark_result['generator_config'])
    output_data.update(element['metrics'])
    yield pd.DataFrame(output_data, index=[sample_id])


@gin.configurable
class SubstructureBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, substruct,
               num_features, hidden_channels, epochs, lr, batch_size):
    self._sample_do_fn = SampleSubstructureDatasetDoFn(param_sampler_specs,
                                                       substruct)
    self._benchmark_par_do = BenchmarkGNNParDo(num_features, hidden_channels, epochs, lr)
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