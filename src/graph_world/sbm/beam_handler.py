import json
import logging
import os

import apache_beam as beam
import gin
import numpy as np

# Change the name of this...

from ..beam.generator_beam_handler import GeneratorBeamHandler
from ..beam.generator_config_sampler import GeneratorConfigSampler
from ..metrics.graph_metrics import GraphMetrics, NodeLabelMetrics
from ..sbm.sbm_simulator import GenerateStochasticBlockModelWithFeatures, MatchType
from ..sbm.utils import sbm_data_to_torchgeo_data, get_kclass_masks, MakePropMat, MakePi
from ..models.benchmarker import BenchmarkGNNParDo


class SampleSbmDoFn(GeneratorConfigSampler, beam.DoFn):

  def __init__(self, param_sampler_specs, marginal=False, normalize_features=True):
    super(SampleSbmDoFn, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('avg_degree', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('p_to_q_ratio', self._SampleUniformFloat)
    self._AddSamplerFn('num_clusters', self._SampleUniformInteger)
    self._AddSamplerFn('cluster_size_slope', self._SampleUniformFloat)
    self._AddSamplerFn('power_exponent', self._SampleUniformFloat)

  def process(self, sample_id):
    """Sample and save SMB outputs given a configuration filepath.
    """
    # Avoid save_main_session in Pipeline args so the controller doesn't
    # have to import the same libraries as the workers which may be using
    # a custom container. The import will execute once then the sys.modeules
    # will be referenced to further calls.

    generator_config, marginal_param, fixed_params = self.SampleConfig(self._marginal)
    generator_config['generator_name'] = 'StochasticBlockModel'

    data = GenerateStochasticBlockModelWithFeatures(
      num_vertices=generator_config['nvertex'],
      num_edges=generator_config['nvertex'] * generator_config['avg_degree'],
      pi=MakePi(generator_config['num_clusters'], generator_config['cluster_size_slope']),
      prop_mat=MakePropMat(generator_config['num_clusters'], generator_config['p_to_q_ratio']),
      num_feature_groups=generator_config['num_clusters'],
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      edge_center_distance=generator_config['edge_center_distance'],
      edge_feature_dim=generator_config['edge_feature_dim'],
      out_degs=np.random.power(generator_config['power_exponent'],
                               generator_config['nvertex']),
      normalize_features=self._normalize_features
    )

    yield {'sample_id': sample_id,
           'marginal_param': marginal_param,
           'fixed_params': fixed_params,
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
    out['metrics'].update(NodeLabelMetrics(element['data'].graph,
                                           element['data'].graph_memberships,
                                           element['data'].node_features))
    yield out


class ConvertToTorchGeoDataParDo(beam.DoFn):
  def __init__(self, output_path, ktrain=5, ktuning=5):
    self._output_path = output_path
    self._ktrain = ktrain
    self._ktuning = ktuning

  def process(self, element):
    sample_id = element['sample_id']
    sbm_data = element['data']

    out = {
      'sample_id': sample_id,
      'metrics' : element['metrics'],
      'torch_data': None,
      'masks': None,
      'skipped': False,
      'generator_config': element['generator_config'],
      'marginal_param': element['marginal_param'],
      'fixed_params': element['fixed_params']
    }

    try:
      torch_data = sbm_data_to_torchgeo_data(sbm_data)
      out['torch_data'] = torch_data
      out['gt_data'] = sbm_data.graph

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
      print(f'failed to convert {sample_id}')
      logging.info(f'Failed to convert sbm_data to torchgeo for sample id {sample_id}')
      yield out
      return

    try:
      out['masks'] = get_kclass_masks(sbm_data, k_train=self._ktrain,
                                      k_val=self._ktuning)

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
      return

    yield out


@gin.configurable
class SbmBeamHandler(GeneratorBeamHandler):

  @gin.configurable
  def __init__(self, param_sampler_specs, benchmarker_wrappers,
               marginal=False, num_tuning_rounds=1,
               tuning_metric='', tuning_metric_is_loss=False, ktrain=5, ktuning=5,
              normalize_features=False):
    self._sample_do_fn = SampleSbmDoFn(param_sampler_specs, marginal, normalize_features)
    self._benchmark_par_do = BenchmarkGNNParDo(benchmarker_wrappers, num_tuning_rounds,
                                               tuning_metric, tuning_metric_is_loss)
    self._metrics_par_do = ComputeSbmGraphMetrics()
    self._ktrain = ktrain
    self._ktuning = ktuning

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
    self._convert_par_do = ConvertToTorchGeoDataParDo(output_path, self._ktrain,
                                                      self._ktuning)
    self._benchmark_par_do.SetOutputPath(output_path)