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
from dataclasses import dataclass, fields
import gin
import numpy as np

from ..beam.generator_config_sampler import GeneratorConfigSampler
from ..generators.sbm_simulator import GenerateStochasticBlockModelWithFeatures, MatchType, MakePi, MakePropMat, MakeDegrees
from ..generators.cabam_simulator import GenerateCABAMGraphWithFeatures
from ..nodeclassification.utils import NodeClassificationDataset


@gin.configurable
class SbmGeneratorWrapper(GeneratorConfigSampler):

  def __init__(self, param_sampler_specs, marginal=False,
               normalize_features=True):
    super(SbmGeneratorWrapper, self).__init__(param_sampler_specs)
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
    self._AddSamplerFn('min_deg', self._SampleUniformInteger)

  def Generate(self, sample_id):
    """Sample and save SMB outputs given a configuration filepath.
    """
    # Avoid save_main_session in Pipeline args so the controller doesn't
    # have to import the same libraries as the workers which may be using
    # a custom container. The import will execute once then the sys.modeules
    # will be referenced to further calls.

    generator_config, marginal_param, fixed_params = self.SampleConfig(
        self._marginal)
    generator_config['generator_name'] = 'StochasticBlockModel'

    sbm_data = GenerateStochasticBlockModelWithFeatures(
      num_vertices=generator_config['nvertex'],
      num_edges=generator_config['nvertex'] * generator_config['avg_degree'],
      pi=MakePi(generator_config['num_clusters'],
                generator_config['cluster_size_slope']),
      prop_mat=MakePropMat(generator_config['num_clusters'],
                           generator_config['p_to_q_ratio']),

      num_feature_groups=generator_config['num_clusters'],
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      edge_center_distance=generator_config['edge_center_distance'],
      edge_feature_dim=generator_config['edge_feature_dim'],
      out_degs=MakeDegrees(generator_config['power_exponent'], 
                               generator_config['min_deg'],
                               generator_config['nvertex']),
      normalize_features=self._normalize_features
    )

    return {'sample_id': sample_id,
            'marginal_param': marginal_param,
            'fixed_params': fixed_params,
            'generator_config': generator_config,
            'data': NodeClassificationDataset(
                graph=sbm_data.graph,
                graph_memberships=sbm_data.graph_memberships,
                node_features=sbm_data.node_features,
                feature_memberships=sbm_data.feature_memberships,
                edge_features=sbm_data.edge_features)}

@gin.configurable
class CABAMGeneratorWrapper(GeneratorConfigSampler):

  def __init__(self, param_sampler_specs, marginal=False,
               normalize_features=False):
    super(CABAMGeneratorWrapper, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('m', self._SampleUniformInteger)
    self._AddSamplerFn('assortativity_type', self._SampleUniformInteger)
    self._AddSamplerFn('inter_link_strength', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('num_clusters', self._SampleUniformInteger)
    self._AddSamplerFn('cluster_size_slope', self._SampleUniformFloat)
    self._AddSamplerFn('temperature', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)


  def Generate(self, sample_id):
    """Sample and save CABAM outputs given a configuration filepath.
    """
    generator_config, marginal_param, fixed_params = self.SampleConfig(
        self._marginal)
    generator_config['generator_name'] = 'CABAM'

    cabam_data = GenerateCABAMGraphWithFeatures(
      n=generator_config['nvertex'],
      m=generator_config['m'],
      inter_link_strength=generator_config['inter_link_strength'],
      num_feature_groups=generator_config['num_clusters'],
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      pi=MakePi(generator_config['num_clusters'],
                generator_config['cluster_size_slope']),
      assortativity_type=generator_config['assortativity_type'],
      temperature=generator_config['temperature'],
      edge_center_distance=generator_config['edge_center_distance'],
      edge_feature_dim=generator_config['edge_feature_dim'],
    )

    return {'sample_id': sample_id,
            'marginal_param': marginal_param,
            'fixed_params': fixed_params,
            'generator_config': generator_config,
            'data': NodeClassificationDataset(
                graph=cabam_data.graph,
                graph_memberships=cabam_data.graph_memberships,
                node_features=cabam_data.node_features,
                feature_memberships=cabam_data.feature_memberships,
                edge_features=cabam_data.edge_features)}