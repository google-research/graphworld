import logging

import apache_beam as beam
import gin
import pandas as pd
from torch_geometric.data import DataLoader

from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..beam.generator_config_sampler import GeneratorConfigSampler
from ..models.benchmarker import BenchmarkGNNParDo
from .simulator import GenerateSubstructureDataset, GetSubstructureGraph
from .utils import substructure_graph_to_torchgeo_data
from ..metrics.graph_metrics import GraphMetrics

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


@gin.configurable
class SubstructureBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, substruct, benchmarker_wrappers, batch_size):
    self._sample_do_fn = SampleSubstructureDatasetDoFn(param_sampler_specs,
                                                       substruct)
    self._benchmark_par_do = BenchmarkGNNParDo(benchmarker_wrappers)
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