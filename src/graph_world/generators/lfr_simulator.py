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
import networkit as nk

from graph_tool.all import *
from graph_world.generators.sbm_simulator import SimulateFeatures, MatchType, SimulateEdgeFeatures

@dataclasses.dataclass
class LFR:
    """
    Stores data for the LFR Model.
    """
    graph: graph_tool.Graph = Ellipsis 
    graph_memberships: np.ndarray = Ellipsis 
    node_features: np.ndarray = Ellipsis 
    feature_memberships: np.ndarray = Ellipsis 
    edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis 


def _GenerateNodeMemberships(n, community_sizes):
  """
  Generates Node Memberships for LFR, based on community sizes from LFR simulation.
  Args:
    n: number of nodes in graph
    community_sizes: 
  Returns:
    memberships: np vector of ints representing community indices
  """
  memberships = np.zeros(n, dtype=int)
  node = 0
  for i in range(len(community_sizes)): 
    memberships[range(node, node + community_sizes[i])] = i 
    node += community_sizes[i]
  return memberships


def NetworkitToGraphWorldData(G, lfr_data, community_sizes):
  """
  Converts NetworKit graph data to GraphWorld LFR dataclass, with node memberships.
  Args:
    G: Networkit generated LFR graph
    lfr_data: LFR dataclass instance to store graph data
    community_sizes: 
  Returns:
    lfr_data: LFR dataclass instance to store graph data
  """  
  nk_edges = list(G.iterEdges())
  nk_nodes = list(G.iterNodes())
  lfr_data.graph = graph_tool.Graph(directed=False)
  vertices = {} 
  for node in nk_nodes:
      v = lfr_data.graph.add_vertex()
      vertices[node] = v
  for src, dst in nk_edges:
      e = lfr_data.graph.add_edge(vertices[src], vertices[dst])
  lfr_data.graph_memberships = _GenerateNodeMemberships(len(nk_nodes), community_sizes)
  return lfr_data

def SimulateLFR(n,
                avg_deg,
                max_deg,
                exponent,
                min_community_size,
                max_community_size,
                exponent_community,
                mu):
  """
  Simulates an LFR Graph using NetworKit and the sampled parameters. 
  Args:
    n: number of nodes 
    avg_deg: average degree 
    max_deg: maximum degree 
    exponent: power law exponent that the degree sequence should follow
    min_community_size: minimum community size 
    max_community_size: maximum community size 
    exponent_community: power law exponent that the community sizes should follow
    mu: mixing parameter
  Returns:
    lfrG: generated NetworKit LFR graph
    community_sizes: sequence of community sizes in generated graph
  """
  for i in range(3):
    try:
      lfr = nk.generators.LFRGenerator(n)
      lfr.generatePowerlawDegreeSequence(avg_deg, 
                                        max_deg, 
                                        exponent)
      lfr.generatePowerlawCommunitySizeSequence(min_community_size, 
                                                max_community_size, 
                                                exponent_community)
      lfr.setMu(mu)
      lfrG = lfr.generate()
      community_sizes = lfr.getPartition().subsetSizes()
      return lfrG, community_sizes
    except Exception:
      print(f'LFR graph not realizeable.. attempt: {i+1}')
      continue
  raise RuntimeWarning(f'LFR graph not realizeable on given parameters.')


def GenerateLFRGraphWithFeatures(n,
                                 avg_deg,
                                 max_deg,
                                 exponent,
                                 min_community_size,
                                 max_community_size,
                                 community_exponent,
                                 mixing_param,
                                 feature_center_distance=0.0,
                                 feature_dim=0,
                                 feature_group_match_type=MatchType.GROUPED,
                                 feature_cluster_variance=1.0,
                                 edge_feature_dim=0,
                                 edge_center_distance=0.0,
                                 edge_cluster_variance=1.0,
                                 normalize_features=True):
  """
  Generates LFR graph for GraphWorld with node and edge features.
  Args:
      n: number of nodes in graph
      avg_deg: average degree 
      max_deg: maximum degree 
      exponent: power law exponent that the degree sequence should follow
      min_community_size: minimum community size 
      max_community_size: maximum community size 
      exponent_community: power law exponent that the community sizes should follow
      mixing_param: mixing parameter
      feature_center_distance: distance between feature cluster centers. When this
          is 0.0, the signal-to-noise ratio is 0.0. When equal to
          feature_cluster_variance, SNR is 1.0.
      feature_dim: dimension of node features.
      feature_group_match_type: see sbm_simulator.MatchType.
      feature_cluster_variance: variance of feature clusters around their centers.
          centers. Increasing this weakens node feature signal.
      edge_feature_dim: dimension of edge features.
      edge_center_distance: per-dimension distance between the intra-class and
        inter-class means. Increasing this strengthens the edge feature signal.
      edge_cluster_variance: variance of edge clusters around their centers.
        Increasing this weakens the edge feature signal.
  Returns:
      result: LFR dataclass instance to store graph data
  """
  result = LFR()
  lfr_model, community_sizes = SimulateLFR(n, 
                                            avg_deg,
                                            max_deg,
                                            exponent,
                                            min_community_size,
                                            max_community_size,
                                            community_exponent,
                                            mixing_param)     
  NetworkitToGraphWorldData(lfr_model, result, community_sizes)
  SimulateFeatures(result, 
                    feature_center_distance,
                    feature_dim,
                    len(community_sizes),
                    feature_group_match_type,
                    feature_cluster_variance,
                    normalize_features)
  SimulateEdgeFeatures(result, 
                        edge_feature_dim,
                        edge_center_distance,
                        edge_cluster_variance)
  return result