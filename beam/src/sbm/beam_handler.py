
import argparse
import json
import logging
import os
import setuptools

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.dataframe.convert import to_dataframe
import gin
import numpy as np
import pandas as pd

# Change the name of this...
from generator_beam_handler import GeneratorBeamHandler
from generator_config_sampler import GeneratorConfigSampler, ParamSamplerSpec
from models.wrappers import LinearGCN
from sbm.sbm_simulator import GenerateStochasticBlockModelWithFeatures, MatchType
from sbm.utils import sbm_data_to_torchgeo_data, get_kclass_masks
from graph_metrics import GraphMetrics

class SampleSbmDoFn(GeneratorConfigSampler, beam.DoFn):

  def __init__(self, param_sampler_specs):
    super(SampleSbmDoFn, self).__init__(param_sampler_specs)
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('avg_degree', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('p_to_q_ratio', self._SampleUniformFloat)

  def process(self, sample_id):
    """Sample and save SMB outputs given a configuration filepath.
    """
    # Avoid save_main_session in Pipeline args so the controller doesn't
    # have to import the same libraries as the workers which may be using
    # a custom container. The import will execute once then the sys.modeules
    # will be referenced to further calls.

    generator_config = self.SampleConfig()
    generator_config['generator_name'] = 'StochasticBlockModel'

    data = GenerateStochasticBlockModelWithFeatures(
      num_vertices=generator_config['nvertex'],
      num_edges=generator_config['nvertex'] * generator_config['avg_degree'],
      pi=np.array([0.25, 0.25, 0.25, 0.25]),
      prop_mat=np.ones((4, 4)) + generator_config['p_to_q_ratio'] * np.diag([1, 1, 1, 1]),
      num_feature_groups=4,
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      edge_center_distance=generator_config['edge_center_distance'],
      edge_feature_dim=generator_config['edge_feature_dim']
    )

    yield {'sample_id': sample_id,
           'generator_config': generator_config,
           'data': data}


class WriteSbmDoFn(beam.DoFn):

  def __init__(self, output_path):
    self._output_path = output_path

  def process(self, element):
    sample_id = element['sample_id']
    config = element['generator_config']
    data = element['data']

    text_mime = 'text/plain'
    prefix = '{0:05}'.format(sample_id)
    config_object_name = os.path.join(self._output_path, prefix + '_config.txt')
    with beam.io.filesystems.FileSystems.create(config_object_name, text_mime) as f:
      buf = bytes(json.dumps(config), 'utf-8')
      f.write(buf)
      f.close()

    graph_object_name = os.path.join(self._output_path, prefix + '_graph.gt')
    with beam.io.filesystems.FileSystems.create(graph_object_name) as f:
      data.graph.save(f)
      f.close()

    graph_memberships_object_name = os.path.join(
      self._output_path, prefix + '_graph_memberships.txt')
    with beam.io.filesystems.FileSystems.create(graph_memberships_object_name, text_mime) as f:
      np.savetxt(f, data.graph_memberships)
      f.close()

    node_features_object_name = os.path.join(
      self._output_path, prefix + '_node_features.txt')
    with beam.io.filesystems.FileSystems.create(node_features_object_name, text_mime) as f:
      np.savetxt(f, data.node_features)
      f.close()

    feature_memberships_object_name = os.path.join(
      self._output_path, prefix + '_feature_membership.txt')
    with beam.io.filesystems.FileSystems.create(feature_memberships_object_name, text_mime) as f:
      np.savetxt(f, data.feature_memberships)
      f.close()

    edge_features_object_name = os.path.join(
      self._output_path, prefix + '_edge_features.txt')
    with beam.io.filesystems.FileSystems.create(edge_features_object_name, text_mime) as f:
      for edge_tuple, features in data.edge_features.items():
        buf = bytes('{0},{1},{2}'.format(edge_tuple[0], edge_tuple[1], features), 'utf-8')
        f.write(buf)
      f.close()


class ComputeSbmGraphMetrics(beam.DoFn):

  def process(self, element):
    out = element
    out['metrics'] = GraphMetrics(element['data'].graph)
    yield out


class ConvertToTorchGeoDataParDo(beam.DoFn):
  def __init__(self, output_path):
    self._output_path = output_path

  def process(self, element):
    sample_id = element['sample_id']
    sbm_data = element['data']

    out = {
      'sample_id': sample_id,
      'metrics' : element['metrics'],
      'torch_data': None,
      'masks': None,
      'skipped': False
    }

    try:
      torch_data = sbm_data_to_torchgeo_data(sbm_data)
      out['torch_data'] = sbm_data_to_torchgeo_data(sbm_data)
      out['generator_config'] = element['generator_config']

      torchgeo_stats = {
        'nodes': torch_data.num_nodes,
        'edges': torch_data.num_edges,
        'average_node_degree': torch_data.num_edges / torch_data.num_nodes,
        # 'contains_isolated_nodes': torchgeo_data.contains_isolated_nodes(),
        # 'contains_self_loops': torchgeo_data.contains_self_loops(),
        # 'undirected': bool(torchgeo_data.is_undirected())
      }
      stats_object_name = os.path.join(self._output_path, '{0:05}_torch_stats.txt'.format(sample_id))
      with beam.io.filesystems.FileSystems.create(stats_object_name, 'text/plain') as f:
        buf = bytes(json.dumps(torchgeo_stats), 'utf-8')
        f.write(buf)
        f.close()
    except:
      out['skipped'] = True
      print(f'faied convert {sample_id}')
      logging.info(f'Failed to convert sbm_data to torchgeo for sample id {sample_id}')
      yield out

    try:
      out['masks'] = get_kclass_masks(sbm_data)

      masks_object_name = os.path.join(self._output_path, '{0:05}_masks.txt'.format(sample_id))
      with beam.io.filesystems.FileSystems.create(masks_object_name, 'text/plain') as f:
        for mask in out['masks']:
          np.savetxt(f, np.atleast_2d(mask.numpy()), fmt='%i', delimiter=' ')
        f.close()
    except:
      out['skipped'] = True
      print(f'failed masks {sample_id}')
      logging.info(f'Failed to sample masks for sample id {sample_id}')
      yield out

    yield out


class BenchmarkGNNParDo(beam.DoFn):
  def __init__(self, num_features, num_classes, hidden_channels, epochs):
    self._num_features = num_features
    self._num_classes = num_classes
    self._hidden_channels = hidden_channels
    self._epochs = epochs

  def SetOutputPath(self, output_path):
    self._output_path = output_path

  def process(self, element):
    sample_id = element['sample_id']
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']

    out = {
      'skipped': skipped,
      'results': None
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return

    train_mask, val_mask, test_mask = masks
    linear_model = LinearGCN(
      self._num_features,
      self._num_classes,
      self._hidden_channels,
      train_mask,
      val_mask,
      self._epochs)

    losses = linear_model.train(torch_data)
    test_accuracy = None
    try:
      # Divide by zero somesimtes happens with the ksample masks.
      test_accuracy = linear_model.test(torch_data)
    except:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')

    benchmark_result = {
      'sample_id': sample_id,
      'losses': losses,
      'test_accuracy': test_accuracy,
      'generator_config': element['generator_config']
    }

    results_object_name = os.path.join(self._output_path, '{0:05}_results.txt'.format(sample_id))
    with beam.io.filesystems.FileSystems.create(results_object_name, 'text/plain') as f:
      buf = bytes(json.dumps(benchmark_result), 'utf-8')
      f.write(buf)
      f.close()

    test_accuracy = (0.0 if benchmark_result['test_accuracy'] is None else
                     benchmark_result['test_accuracy'])
    output_data = {"test_accuracy": test_accuracy}
    output_data.update(benchmark_result['generator_config'])
    output_data.update(element['metrics'])
    yield pd.DataFrame(output_data, index=[sample_id])


@gin.configurable
class SbmBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs,
               num_features, num_classes, hidden_channels, epochs):
    self._sample_do_fn = SampleSbmDoFn(param_sampler_specs)
    self._benchmark_par_do = BenchmarkGNNParDo(num_features, num_classes, hidden_channels, epochs)
    self._metrics_par_do = ComputeSbmGraphMetrics()


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
    self._convert_par_do = ConvertToTorchGeoDataParDo(output_path)
    self._benchmark_par_do.SetOutputPath(output_path)