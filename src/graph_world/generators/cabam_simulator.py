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
import networkx as nx
from tqdm.notebook import tqdm
from cabam import CABAM as CABAM_git

from graph_tool.all import *
from graph_world.generators.sbm_simulator import SimulateFeatures, MatchType, SimulateEdgeFeatures

@dataclasses.dataclass
class CABAM:
    """
    Stores data for Class Assortative and Attributed graphs via the Barabasi Albert Model. Identical to SBM dataclass.
    """
    graph: graph_tool.Graph = Ellipsis
    graph_memberships: np.ndarray = Ellipsis
    node_features: np.ndarray = Ellipsis
    feature_memberships: np.ndarray = Ellipsis
    edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis


def NetworkxToGraphWorldData(G, node_labels, cabam_data):
  """
  Converts NetworkX graph data to GraphWorld CABAM dataclass.
  Args:
    G: NetworkX Graph 
    node_labels: list of integer node labels
    cabam_data: CABAM dataclass instance to store graph data
  Returns:
    cabam_data
  """
  cabam_data.graph_memberships = list(node_labels) # Memberships is integer node class list

  # Manipulate G into cabam_data.graph Graph Tool object
  nx_edges = list(G.edges())
  nx_nodes = list(G.nodes())
  cabam_data.graph = graph_tool.Graph(directed=False)

  # Add the nodes
  vertices = {} # vertex mapping for tracking edges later
  for node in nx_nodes:
      # Create the vertex and annotate for our edges later
      v = cabam_data.graph.add_vertex()
      vertices[node] = v
  # Add the edges
  for src, dst in nx_edges:
      # Look up the vertex structs from our vertices mapping and add edge.
      e = cabam_data.graph.add_edge(vertices[src], vertices[dst])
  return cabam_data


def GenerateAssortativityDict(p_in, assortativity_type, temperature):
    """
    Generates a dictionary representing the Assortativity Constant in CABAM generation - the parameter named 'c_probs'.
    Args:
        p_in: float representing probability of intra-class assignment in CABAM generation with FIXED assortativity
        assortativity_type: integer representing assortativity type chosen
        temperature: integer representing temperature of tanh function in CABAM generation with DEGREE DEPENDENT assortativity        
    """
    if assortativity_type==1: # Fixed assortativity 
      return {1: p_in, 0: 1-p_in }
    if assortativity_type==2: # Degree dependent assortativity
      return lambda k: {1: np.tanh(k/temperature), 0: 1 - np.tanh(k/temperature)}


def GenerateCABAMGraphWithFeatures(
    n,
    m,
    inter_link_strength,
    pi,
    assortativity_type,
    temperature,
    feature_center_distance=0.0,
    feature_dim=0,
    num_feature_groups=1,
    feature_group_match_type=MatchType.RANDOM,
    feature_cluster_variance=1.0,
    edge_feature_dim=0,
    edge_center_distance=0.0,
    edge_cluster_variance=1.0,
    normalize_features=True):
    """
    Generates Class Assortative Graphs via the Barabasi Albert Model (CABAM) with node features.
    Args:
        n: number of nodes in graph
        m: number of edges to add at each timestep in graph generation
        p_in: float representing probability of intra-class assignment in CABAM generation with FIXED assortativity
        pi: class assignment probability vector
        assortativity_type: integer representing assortativity type chosen
        temperature: integer representing temperature of tanh function in CABAM generation with DEGREE DEPENDENT assortativity
        feature_center_distance: distance between feature cluster centers. When this
            is 0.0, the signal-to-noise ratio is 0.0. When equal to
            feature_cluster_variance, SNR is 1.0.
        feature_dim: dimension of node features.
        num_feature_groups: number of feature clusters.
        feature_group_match_type: see sbm_simulator.MatchType.
        feature_cluster_variance: variance of feature clusters around their centers.
            centers. Increasing this weakens node feature signal.
    Returns:
        result: CABAM dataclass instance to store graph data
    """
    result = CABAM()
    CABAM_model = CABAM_git()
    G, _, node_labels, _, _ = CABAM_model.generate_graph(n=n, m=m, num_classes=num_feature_groups, native_class_probs=pi.tolist(), inter_intra_link_probs=GenerateAssortativityDict(inter_link_strength, assortativity_type, temperature) )
    NetworkxToGraphWorldData(G, node_labels, result)

    # Borrowing node and edge feature generation from SBM
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
