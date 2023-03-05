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

import collections
import enum
import math
import random
from typing import Dict, Sequence, List, Tuple
from sklearn.preprocessing import normalize

import dataclasses
import graph_tool
import numpy as np

from graph_tool.all import *


class MatchType(enum.Enum):
  """Indicates type of feature/graph membership matching to do.
    RANDOM: feature memberships are generated randomly.
    NESTED: for # feature groups >= # graph groups. Each feature cluster is a
      sub-cluster of a graph cluster. Multiplicity of sub-clusters per
      graph cluster is kept as uniform as possible.
    GROUPED: for # feature groups <= # graph groups. Each graph cluster is a
      sub-cluster of a feature cluster. Multiplicity of sub-clusters per
      feature cluster is kept as uniform as possible.
  """
  RANDOM = 1
  NESTED = 2
  GROUPED = 3


@dataclasses.dataclass
class StochasticBlockModel:
  """Stores data for stochastic block model graphs with features.
  Attributes:
    graph: graph-tool Graph object.
    graph_memberships: list of integer node classes.
    node_features: numpy array of node features.
    feature_memberships: list of integer node feature classes.
    edge_features: map from edge tuple to numpy array. Only stores undirected
      edges, i.e. (0, 1) will be in the map, but (1, 0) will not be.
  """
  graph: graph_tool.Graph = Ellipsis
  graph_memberships: np.ndarray = Ellipsis
  node_features: np.ndarray = Ellipsis
  feature_memberships: np.ndarray = Ellipsis
  edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis


def _GetNestingMap(large_k, small_k):
  """Given two group sizes, computes a "nesting map" between groups.
  This function will produce a bipartite map between two sets of "group nodes"
  that will be used downstream to partition nodes in a bigger graph. The map
  encodes which groups from the larger set are nested in certain groups from
  the smaller set.
  As currently implemented, nesting is assigned as evenly as possible. If
  large_k is an integer multiple of small_k, each smaller-set group will be
  mapped to exactly (large_k/small_k) larger-set groups. If there is a
  remainder r, the first r smaller-set groups will each have one extra nested
  larger-set group.
  Args:
    large_k: (int) size of the larger group set
    small_k: (int) size of the smaller group set
  Returns:
    nesting_map: (dict) map from larger group set indices to lists of
      smaller group set indices
  """
  min_multiplicity = int(math.floor(large_k / small_k))
  max_bloated_group_index = large_k - small_k * min_multiplicity - 1
  nesting_map = collections.defaultdict(list)
  pos = 0
  for i in range(small_k):
    for _ in range(min_multiplicity + int(i <= max_bloated_group_index)):
      nesting_map[i].append(pos)
      pos += 1
  return nesting_map


def _GenerateFeatureMemberships(
    graph_memberships,
    num_groups=None,
    match_type=MatchType.RANDOM):
  """Generates a feature membership assignment.
  Args:
    graph_memberships: (list) the integer memberships for the graph SBM
    num_groups: (int) number of groups. If None, defaults to number of unique
      values in graph_memberships.
    match_type: (MatchType) see the enum class description.
  Returns:
    memberships: a int list - index i contains feature group of node i.
  """
  # Parameter checks
  if num_groups is not None and num_groups == 0:
    raise ValueError("argument num_groups must be None or positive")
  graph_num_groups = len(set(graph_memberships))
  if num_groups is None:
    num_groups = graph_num_groups

  # Compute memberships
  memberships = []
  if match_type == MatchType.GROUPED:
    if num_groups > graph_num_groups:
      raise ValueError(
        "for match type GROUPED, must have num_groups <= graph_num_groups")
    nesting_map = _GetNestingMap(graph_num_groups, num_groups)
    # Creates deterministic map from (smaller) graph clusters to (larger)
    # feature clusters.
    reverse_nesting_map = {}
    for feature_cluster, graph_cluster_list in nesting_map.items():
      for cluster in graph_cluster_list:
        reverse_nesting_map[cluster] = feature_cluster
    for cluster in graph_memberships:
      memberships.append(reverse_nesting_map[cluster])
  elif match_type == MatchType.NESTED:
    if num_groups < graph_num_groups:
      raise ValueError(
        "for match type NESTED, must have num_groups >= graph_num_groups")
    nesting_map = _GetNestingMap(num_groups, graph_num_groups)
    # Creates deterministic map from (smaller) feature clusters to (larger)
    # graph clusters.
    for graph_cluster_id, feature_cluster_ids in nesting_map.items():
      sorted_feature_cluster_ids = sorted(feature_cluster_ids)
      num_feature_groups = len(sorted_feature_cluster_ids)
      feature_pi = np.ones(num_feature_groups) / num_feature_groups
      num_graph_cluster_nodes = np.sum(
        [i == graph_cluster_id for i in graph_memberships])
      sub_memberships = _GenerateNodeMemberships(num_graph_cluster_nodes,
                                                 feature_pi)
      sub_memberships = [sorted_feature_cluster_ids[i] for i in sub_memberships]
      memberships.extend(sub_memberships)
  else:  # MatchType.RANDOM
    memberships = random.choices(range(num_groups), k=len(graph_memberships))
  return np.array(sorted(memberships))


def _ComputeExpectedEdgeCounts(num_edges, num_vertices,
                               pi,
                               prop_mat):
  """Computes expected edge counts within and between communities.
  Args:
    num_edges: expected number of edges in the graph.
    num_vertices: number of nodes in the graph
    pi: interable of non-zero community size proportions. Must sum to 1.0, but
      this check is left to the caller of this internal function.
    prop_mat: square, symmetric matrix of community edge count rates. Entries
      must be non-negative, but this check is left to the caller.
  Returns:
    symmetric matrix with shape prop_mat.shape giving expected edge counts.
  """
  scale = np.matmul(pi, np.matmul(prop_mat, pi)) * num_vertices ** 2
  prob_mat = prop_mat * num_edges / scale
  return np.outer(pi, pi) * prob_mat * num_vertices ** 2


def _ComputeCommunitySizes(num_vertices, pi):
  """Helper function of GenerateNodeMemberships to compute group sizes.
  Args:
    num_vertices: number of nodes in graph.
    pi: interable of non-zero community size proportions.
  Returns:
    community_sizes: np vector of group sizes. If num_vertices * pi[i] is a
      whole number (up to machine precision), community_sizes[i] will be that
      number. Otherwise, this function accounts for rounding errors by making
      group sizes as balanced as possible (i.e. increasing smallest groups by
      1 or decreasing largest groups by 1 if needed).
  """
  community_sizes = [int(x * num_vertices) for x in pi]
  if sum(community_sizes) != num_vertices:
    size_order = np.argsort(community_sizes)
    delta = sum(community_sizes) - num_vertices
    adjustment = np.sign(delta)
    if adjustment == 1:
      size_order = np.flip(size_order)
    for i in range(int(abs(delta))):
      community_sizes[size_order[i]] -= adjustment
  return community_sizes


def _GenerateNodeMemberships(num_vertices,
                             pi):
  """Gets node memberships for sbm.
  Args:
    num_vertices: number of nodes in graph.
    pi: interable of non-zero community size proportions. Must sum to 1.0, but
      this check is left to the caller of this internal function.
  Returns:
    np vector of ints representing community indices.
  """
  community_sizes = _ComputeCommunitySizes(num_vertices, pi)
  memberships = np.zeros(num_vertices, dtype=int)
  node = 0
  for i in range(len(pi)):
    memberships[range(node, node + community_sizes[i])] = i
    node += community_sizes[i]
  return memberships


def SimulateSbm(sbm_data,
                num_vertices,
                num_edges,
                pi,
                prop_mat,
                out_degs=None):
  """Generates a stochastic block model, storing data in sbm_data.graph.
  This function uses graph_tool.generate_sbm. Refer to that
  documentation for more information on the model and parameters.
  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    num_vertices: (int) number of nodes in the graph.
    num_edges: (int) expected number of edges in the graph.
    pi: iterable of non-zero community size proportions. Must sum to 1.0.
    prop_mat: square, symmetric matrix of community edge count rates.
    out_degs: Out-degree propensity for each node. If not provided, a constant
      value will be used. Note that the values will be normalized inside each
      group, if they are not already so.
  Returns: (none)
  """
  if round(abs(np.sum(pi) - 1.0), 12) != 0:
    raise ValueError("entries of pi ( must sum to 1.0")
  if prop_mat.shape[0] != len(pi) or prop_mat.shape[1] != len(pi):
    raise ValueError("prop_mat must be k x k where k = len(pi)")
  sbm_data.graph_memberships = _GenerateNodeMemberships(num_vertices, pi)
  edge_counts = _ComputeExpectedEdgeCounts(num_edges, num_vertices, pi,
                                           prop_mat)
  sbm_data.graph = graph_tool.generation.generate_sbm(
    sbm_data.graph_memberships, edge_counts, out_degs)
  graph_tool.generation.remove_self_loops(sbm_data.graph)
  graph_tool.generation.remove_parallel_edges(sbm_data.graph)
  sbm_data.graph.reindex_edges()


def SimulateFeatures(sbm_data,
                     center_var,
                     feature_dim,
                     num_groups,
                     match_type=MatchType.RANDOM,
                     cluster_var=1.0,
                     normalize_features=True):
  """Generates node features using multivate normal mixture model.
  This function does nothing and throws a warning if
  sbm_data.graph_memberships is empty. Run SimulateSbm to fill that field.
  Feature data is stored as an attribute of sbm_data named 'node_features'.
  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    center_var: (float) variance of feature cluster centers. When this is 0.0,
      the signal-to-noise ratio is 0.0. When equal to cluster_var, SNR is 1.0.
    feature_dim: (int) dimension of the multivariate normal.
   num_groups: (int) number of centers. Generated by a multivariate normal with
     mean zero and covariance matrix cluster_var * I_{feature_dim}.
    match_type: (MatchType) see sbm_simulator.MatchType for details.
    cluster_var: (float) variance of feature clusters around their centers.
  Raises:
    RuntimeWarning: if simulator has no graph or a graph with no nodes.
  """
  if sbm_data.graph_memberships is None:
    raise RuntimeWarning("No graph_memberships found: no features generated. "
                         "Run SimulateSbm to generate graph_memberships.")

  # Get memberships
  sbm_data.feature_memberships = _GenerateFeatureMemberships(
    graph_memberships=sbm_data.graph_memberships,
    num_groups=num_groups,
    match_type=match_type)

  # Get centers
  centers = []
  center_cov = np.identity(feature_dim) * center_var
  cluster_cov = np.identity(feature_dim) * cluster_var
  for _ in range(num_groups):
    center = np.random.multivariate_normal(
      np.zeros(feature_dim), center_cov, 1)[0]
    centers.append(center)
  features = []
  for cluster_index in sbm_data.feature_memberships:
    feature = np.random.multivariate_normal(centers[cluster_index], cluster_cov,
                                            1)[0]
    features.append(feature)
  features = np.array(features)
  if normalize_features:
    features = normalize(features)
  sbm_data.node_features = features


def SimulateEdgeFeatures(sbm_data,
                         feature_dim,
                         center_distance=0.0,
                         cluster_variance=1.0):
  """Generates edge feature distribution via inter-class vs intra-class.
  Edge feature data is stored as an sbm_data attribute named `edge_feature`, a
  dict from 2-tuples of node IDs to numpy vectors.
  Edge features have two centers: one at (0, 0, ....) and one at
  (center_distance, center_distance, ....) for inter-class and intra-class
  edges (respectively). They are generated from a multivariate normal with
  covariance matrix = cluster_variance * I_d.
  Requires non-None `graph` and `graph_memberships` attributes in sbm_data.
  Use SimulateSbm to generate them. Throws warning if either are None.
  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    feature_dim: (int) dimension of the multivariate normal.
    center_distance: (float) per-dimension distance between the intra-class and
      inter-class means. Increasing this makes the edge feature signal stronger.
    cluster_variance: (float) variance of clusters around their centers.
  Raises:
    RuntimeWarning: if simulator has no graph or a graph with no nodes.
  """
  if sbm_data.graph is None:
    raise RuntimeWarning("SbmSimulator has no graph: no features generated.")
  if sbm_data.graph.num_vertices() == 0:
    raise RuntimeWarning("graph has no nodes: no features generated.")
  if sbm_data.graph_memberships is None:
    raise RuntimeWarning("graph has no memberships: no features generated.")

  center0 = np.zeros(shape=(feature_dim,))
  center1 = np.ones(shape=(feature_dim,)) * center_distance
  covariance = np.identity(feature_dim) * cluster_variance
  sbm_data.edge_features = {}
  for edge in sbm_data.graph.edges():
    vertex1 = int(edge.source())
    vertex2 = int(edge.target())
    edge_tuple = tuple(sorted((vertex1, vertex2)))
    if (sbm_data.graph_memberships[vertex1] ==
        sbm_data.graph_memberships[vertex2]):
      center = center1
    else:
      center = center0
    sbm_data.edge_features[edge_tuple] = np.random.multivariate_normal(
      center, covariance, 1)[0]


def GenerateStochasticBlockModelWithFeatures(
    num_vertices,
    num_edges,
    pi,
    prop_mat,
    out_degs=None,
    feature_center_distance=0.0,
    feature_dim=0,
    num_feature_groups=1,
    feature_group_match_type=MatchType.RANDOM,
    feature_cluster_variance=1.0,
    edge_feature_dim=0,
    edge_center_distance=0.0,
    edge_cluster_variance=1.0,
    normalize_features=True):
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
                   feature_dim,
                   num_feature_groups,
                   feature_group_match_type,
                   feature_cluster_variance,
                   normalize_features)
  SimulateEdgeFeatures(result, edge_feature_dim,
                       edge_center_distance,
                       edge_cluster_variance)
  return result


# Helper function to create the "Pi" vector for the SBM model (the
# ${num_communities}-simplex vector giving relative community sizes) from
# the `community_size_slope` config field. See the config proto for details.
def MakePi(num_communities: int, community_size_slope: float) -> np.ndarray:
  pi = np.array(range(num_communities)) * community_size_slope
  pi += np.ones(num_communities)
  pi /= np.sum(pi)
  return pi


# Helper function to create the "PropMat" matrix for the SBM model (square
# matrix giving inter-community Poisson means) from the config parameters,
# particularly `p_to_q_ratio`. See the config proto for details.
def MakePropMat(num_communities: int, p_to_q_ratio: float) -> np.ndarray:
  prop_mat = np.ones((num_communities, num_communities))
  np.fill_diagonal(prop_mat, p_to_q_ratio)
  return prop_mat


# Helper function to create a degree set that follows a power law for the
# 'out_degs' parameter in SBM construction.
def MakeDegrees(power_exponent, min_deg, num_vertices):
  degrees = np.zeros(num_vertices)
  k_min = min_deg
  k_max = num_vertices
  gamma = power_exponent
  for i in range(num_vertices):
      degrees[i] = int(power_law(k_min, k_max, np.random.uniform(0,1), gamma))
  return degrees


# Helper function of MakeDegrees to construct power law samples
def power_law(k_min, k_max, y, gamma):
  return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma + 1.0))
