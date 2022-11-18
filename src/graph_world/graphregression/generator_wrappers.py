# Copyright 2022 Google LLC
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
import enum

import dataclasses
import gin
import graph_tool
import numpy as np
from sklearn.preprocessing import scale

from ..beam.generator_config_sampler import GeneratorConfigSampler
from ..generators.er_simulator import erdos_graph
from ..graphregression.utils import GraphRegressionDataset


def _get_star_graph():
  g = graph_tool.Graph(directed=False)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(0, 3)
  return g


def _get_triangle_graph():
  g = graph_tool.Graph(directed=False)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  g.add_edge(2, 0)
  return g


def _get_tailed_triangle_graph():
  g = _get_triangle_graph()
  g.add_edge(0, 3)
  return g


def _get_chordal_cycle_graph():
  g = _get_tailed_triangle_graph()
  g.add_edge(1, 3)
  return g


@gin.constants_from_enum
class Substructure(enum.Enum):
  """Indicates type of substructure to count.
  """
  STAR_GRAPH = 1
  TRIANGLE_GRAPH = 2
  TAILED_TRIANGLE_GRAPH = 3
  CHORDAL_CYCLE_GRAPH = 4


def _GetSubstructureGraph(substruct: Substructure):
  if substruct == Substructure.STAR_GRAPH:
    return _get_star_graph()
  elif substruct == Substructure.TRIANGLE_GRAPH:
    return _get_triangle_graph()
  elif substruct == Substructure.CHORDAL_CYCLE_GRAPH:
    return _get_chordal_cycle_graph()
  elif substruct == Substructure.TAILED_TRIANGLE_GRAPH:
    return _get_tailed_triangle_graph()


def _GenerateSubstructureDataset(
    num_graphs: int,
    num_vertices: int,
    edge_prob: float,
    substruct_graph: graph_tool.Graph):
  graphs = [erdos_graph(num_vertices, edge_prob) for _ in range(num_graphs)]
  substruct_counts = []
  for graph in graphs:
    assert graph.num_vertices() == num_vertices, "num_vertices is %d" % graph.num_vertices()
    _, counts = graph_tool.clustering.motifs(
      g=graph,
      k=substruct_graph.num_vertices(),
      motif_list=[substruct_graph]
    )
    substruct_counts.append(counts[0])
  return {'graphs': graphs, 'substruct_counts': substruct_counts}


@gin.configurable
class SubstructureGeneratorWrapper(GeneratorConfigSampler):

  def __init__(self, param_sampler_specs, substruct, normalize_target=True,
               marginal=False):
    super(SubstructureGeneratorWrapper, self).__init__(param_sampler_specs)
    self._marginal = marginal
    self._AddSamplerFn('num_graphs', self._SampleUniformInteger)
    self._AddSamplerFn('num_vertices', self._SampleUniformInteger)
    self._AddSamplerFn('edge_prob', self._SampleUniformFloat)
    self._AddSamplerFn('train_prob', self._SampleUniformFloat)
    self._AddSamplerFn('tuning_prob', self._SampleUniformFloat)
    self._substruct = substruct
    self._normalize_target = normalize_target

  def Generate(self, sample_id):
    """Sample substructure dataset.
    """

    generator_config, marginal_param, fixed_params = self.SampleConfig(
        self._marginal)
    generator_config['generator_name'] = 'Substructure'

    data = _GenerateSubstructureDataset(
      num_graphs=generator_config['num_graphs'],
      num_vertices=generator_config['num_vertices'],
      edge_prob=generator_config['edge_prob'],
      substruct_graph=_GetSubstructureGraph(self._substruct)
    )

    if self._normalize_target:
      data['substruct_counts'] = scale(data['substruct_counts'])

    node_features = [
        np.ones(shape=[graph.num_vertices(), 1], dtype=np.float32) for
        graph in data['graphs']]

    return {'sample_id': sample_id,
            'marginal_param': marginal_param,
            'fixed_params': fixed_params,
            'generator_config': generator_config,
            'data': GraphRegressionDataset(
                graphs=data['graphs'],
                graph_node_features=node_features,
                graph_regression_target=data['substruct_counts'])}
