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
from ..generators.lfr_simulator import GenerateLFRGraphWithFeatures, SimulateLFRWrapper
from ..nodeclassification.utils import NodeClassificationDataset


@gin.configurable
class SbmGeneratorWrapper(GeneratorConfigSampler):

  def __init__(self, param_sampler_specs, marginal=False,
               normalize_features=True, use_generated_lfr_communities=False, lfr_params=None):
    super(SbmGeneratorWrapper, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._use_generated_lfr_communities = use_generated_lfr_communities
    self._lfr_params = lfr_params
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
    if use_generated_lfr_communities:
      self._AddLFRParameters()

  def _AddLFRParameters(self):
    # Add and sample only those LFR parameters relevant to community generation 
    lfr_community_param_names = ['community_min_size_proportion','community_max_size_proportion',
                                 'community_power_exponent']
    self._param_sampler_specs = {**{spec.name: spec for spec in self._lfr_params 
                                 if spec.name in lfr_community_param_names}, 
                                 **self._param_sampler_specs}
    self._AddSamplerFn('community_max_size_proportion', self._SampleUniformFloat)
    self._AddSamplerFn('community_min_size_proportion', self._SampleUniformFloat)
    self._AddSamplerFn('community_power_exponent', self._SampleUniformFloat)    

    # Pull remaining parameters for LFR generation from LFR spec
    lfr_other_param_names = ['power_exponent', 'avg_degree', 'mixing_param', 'max_degree_proportion']
    self._lfr_params = {spec.name: spec for spec in self._lfr_params if spec.name in lfr_other_param_names}

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

    if self._use_generated_lfr_communities: 
      _, lfr_model = SimulateLFRWrapper(generator_config['nvertex'], 
                                        self._lfr_params['avg_degree'].default_val, 
                                        int(self._lfr_params['max_degree_proportion'].default_val*generator_config['nvertex']),
                                        self._lfr_params['power_exponent'].default_val,
                                        int(generator_config['community_min_size_proportion']*generator_config['nvertex']), 
                                        int(generator_config['community_max_size_proportion']*generator_config['nvertex']), 
                                        generator_config['community_power_exponent'], 
                                        self._lfr_params['mixing_param'].default_val) 
      if lfr_model is None:
        return {'sample_id': sample_id,
                'marginal_param': marginal_param,
                'fixed_params': fixed_params,
                'generator_config': generator_config,
                'data': None}

      communities = lfr_model.getPartition().subsetSizes()
      generator_config['num_clusters'] = len(communities)
      pi = np.array(communities) / sum(communities)
    else:
      pi=MakePi(generator_config['num_clusters'],generator_config['cluster_size_slope'])
      
    prop_mat=MakePropMat(generator_config['num_clusters'],generator_config['p_to_q_ratio'])
    sbm_data = GenerateStochasticBlockModelWithFeatures(
      num_vertices=generator_config['nvertex'],
      num_edges=generator_config['nvertex'] * generator_config['avg_degree'],
      pi=pi,
      prop_mat=prop_mat,
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
               normalize_features=False, use_generated_lfr_communities=False, lfr_params=None):
    super(CABAMGeneratorWrapper, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._use_generated_lfr_communities = use_generated_lfr_communities
    self._lfr_params = lfr_params
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('min_deg', self._SampleUniformInteger)
    self._AddSamplerFn('assortativity_type', self._SampleUniformInteger)
    self._AddSamplerFn('inter_link_strength', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('num_clusters', self._SampleUniformInteger)
    self._AddSamplerFn('cluster_size_slope', self._SampleUniformFloat)
    self._AddSamplerFn('temperature', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)
    if use_generated_lfr_communities:
      SbmGeneratorWrapper._AddLFRParameters(self)

  def Generate(self, sample_id):
    """Sample and save CABAM outputs given a configuration filepath.
    """
    generator_config, marginal_param, fixed_params = self.SampleConfig(
        self._marginal)
    generator_config['generator_name'] = 'CABAM'

    if self._use_generated_lfr_communities:
      _, lfr_model = SimulateLFRWrapper(generator_config['nvertex'], 
                                        self._lfr_params['avg_degree'].default_val, 
                                        int(self._lfr_params['max_degree_proportion'].default_val*generator_config['nvertex']),
                                        self._lfr_params['power_exponent'].default_val,
                                        int(generator_config['community_min_size_proportion']*generator_config['nvertex']), 
                                        int(generator_config['community_max_size_proportion']*generator_config['nvertex']), 
                                        generator_config['community_power_exponent'], 
                                        self._lfr_params['mixing_param'].default_val)                                 
      if lfr_model is None:
        return {'sample_id': sample_id,
                'marginal_param': marginal_param,
                'fixed_params': fixed_params,
                'generator_config': generator_config,
                'data': None}
                
      communities = lfr_model.getPartition().subsetSizes()
      generator_config['num_clusters'] = len(communities)
      pi = np.array(communities) / sum(communities)
    else:
      pi=MakePi(generator_config['num_clusters'],generator_config['cluster_size_slope'])

    cabam_data = GenerateCABAMGraphWithFeatures(
      n=generator_config['nvertex'],
      min_deg=generator_config['min_deg'],
      inter_link_strength=generator_config['inter_link_strength'],
      num_feature_groups=generator_config['num_clusters'],
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      pi=pi,
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


@gin.configurable
class LFRGeneratorWrapper(GeneratorConfigSampler):

  def __init__(self, param_sampler_specs, marginal=False,
               normalize_features=True, num_tries=20):
    super(LFRGeneratorWrapper, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._normalize_features = normalize_features
    self._num_tries = num_tries
    self._AddSamplerFn('nvertex', self._SampleUniformInteger)
    self._AddSamplerFn('avg_degree', self._SampleUniformFloat)
    self._AddSamplerFn('power_exponent', self._SampleUniformFloat)
    self._AddSamplerFn('mixing_param', self._SampleUniformFloat)
    self._AddSamplerFn('max_degree_proportion', self._SampleUniformFloat)
    self._AddSamplerFn('community_max_size_proportion', self._SampleUniformFloat)
    self._AddSamplerFn('community_min_size_proportion', self._SampleUniformFloat)
    self._AddSamplerFn('community_power_exponent', self._SampleUniformFloat)
    self._AddSamplerFn('feature_center_distance', self._SampleUniformFloat)
    self._AddSamplerFn('feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_feature_dim', self._SampleUniformInteger)
    self._AddSamplerFn('edge_center_distance', self._SampleUniformFloat)

  def Generate(self, sample_id):
    """
    Sample and save LFR outputs given a configuration filepath.
    """
    generator_config, marginal_param, fixed_params = self.SampleConfig(
        self._marginal)
    generator_config['generator_name'] = 'LFR'

    lfr_data = GenerateLFRGraphWithFeatures(
      n=generator_config['nvertex'],
      avg_deg=generator_config['avg_degree'],
      max_deg=int(generator_config['max_degree_proportion']*generator_config['nvertex']),
      exponent=generator_config['power_exponent'],
      min_community_size=int(generator_config['community_min_size_proportion']*generator_config['nvertex']),
      max_community_size=int(generator_config['community_max_size_proportion']*generator_config['nvertex']),
      community_exponent=generator_config['community_power_exponent'],
      mixing_param=generator_config['mixing_param'],
      feature_group_match_type=MatchType.GROUPED,
      feature_center_distance=generator_config['feature_center_distance'],
      feature_dim=generator_config['feature_dim'],
      edge_center_distance=generator_config['edge_center_distance'],
      edge_feature_dim=generator_config['edge_feature_dim'],
      normalize_features=self._normalize_features,
      num_tries=self._num_tries
    )

    if lfr_data:
      data=NodeClassificationDataset(
                graph=lfr_data.graph,
                graph_memberships=lfr_data.graph_memberships,
                node_features=lfr_data.node_features,
                feature_memberships=lfr_data.feature_memberships,
                edge_features=lfr_data.edge_features)
    else: data=None

    return {'sample_id': sample_id,
            'marginal_param': marginal_param,
            'fixed_params': fixed_params,
            'generator_config': generator_config,
            'data': data}