# Copyright 2021 Google LLC
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

#@title Wrapper code for SBM generator class
# pasted from https://github.com/google-research/google-research/blob/master/graph_embedding/simulations/simulations.py
import typing

import numpy as np

from smb.smb_simulator import SimulateEdgeFeatures

# Types
List = typing.List

def GenerateStochasticBlockModelWithFeatures(
    num_vertices,
    num_edges,
    pi,
    prop_mat,
    out_degs = None,
    feature_center_distance = 0.0,
    feature_dim = 0,
    num_feature_groups = 1,
    feature_group_match_type = MatchType.RANDOM,
    feature_cluster_variance = 1.0,
    edge_feature_dim = 0,
    edge_center_distance = 0.0,
    edge_cluster_variance = 1.0):
  """Generates stochastic block model (SBM) with node features.
  Args:
    num_vertices: number of nodes in the graph.
    num_edges: expected number of edges in the graph.
    pi: interable of non-zero community size proportions. Must sum to 1.0.
    prop_mat: square, symmetric matrix of community edge count rates. Example:
      if diagonals are 2.0 and off-diagonals are 1.0, within-community edges are
      twices as likely as between-community edges.
    out_degs: Out-degree propensity for each node. If not provided, a constant
      value will be used. Note that the values will be normalized inside each
      group, if they are not already so.
    feature_center_distance: distance between feature cluster centers. When this
      is 0.0, the signal-to-noise ratio is 0.0. When equal to
      feature_cluster_variance, SNR is 1.0.
    feature_dim: dimension of node features.
    num_feature_groups: number of feature clusters.
    feature_group_match_type: see sbm_simulator.MatchType.
    feature_cluster_variance: variance of feature clusters around their centers.
      centers. Increasing this weakens node feature signal.
    edge_feature_dim: dimension of edge features.
    edge_center_distance: per-dimension distance between the intra-class and
      inter-class means. Increasing this strengthens the edge feature signal.
    edge_cluster_variance: variance of edge clusters around their centers.
      Increasing this weakens the edge feature signal.
  Returns:
    result: a StochasticBlockModel data class.
  """
  result = StochasticBlockModel()
  SimulateSbm(result, num_vertices, num_edges, pi, prop_mat, out_degs)
  SimulateFeatures(result, feature_center_distance,
                   feature_dim, num_feature_groups,
                   feature_group_match_type, feature_cluster_variance)
  SimulateEdgeFeatures(result, edge_feature_dim,
                       edge_center_distance,
                       edge_cluster_variance)
  return result
