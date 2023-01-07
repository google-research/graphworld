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
import logging
from tqdm.notebook import tqdm


from graph_tool.all import *
from graph_world.generators.sbm_simulator import SimulateFeatures, MatchType
    #SBM class called within Generate method, we may only need the helper functions to create the features
    #if we need entire generate method, modify to make this file just use SBM class 
    #if method needs modifications, extend SBM class 

#from sbm_simulator import StochasticBlockModel

class AssortativityType(enum.Enum):
  """Indicates type of assortativity to use in graph generation.
    FIXED: 
    VARIABLE:  
  """
  FIXED = 1
  VARIABLE = 2


@dataclasses.dataclass
class CABAM:
    """Stores data for Class Assortative and Attributed graphs via the Barabasi Albert Model.
    Attributes:

      !!!Identical and equal to all _task_Dataset classes!!!

      graph: graph-tool Graph object.
      graph_memberships: list of integer node classes.
      node_features: numpy array of node features.
      feature_memberships: list of integer node feature classes.
      edge_features: map from edge tuple to numpy array. Only stores undirected
        edges, i.e. (0, 1) will be in the map, but (1, 0) will not be.
    """
    graph: graph_tool.Graph = Ellipsis # ok - converted graph from CABAM
    graph_memberships: np.ndarray = Ellipsis # ok - node labels from CABAM
    node_features: np.ndarray = Ellipsis # GENERATE
    feature_memberships: np.ndarray = Ellipsis # GENERATE
    edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis # LEAVE EMPTY
    
def NetworkxToGraphWorldData(G, node_labels, cabam_data):
  cabam_data.graph_memberships = list(node_labels) # memberships is integer node class list

  # Manipulate G into cabam_data.graph gt object
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

    # null the edge features
  cabam_data.edge_features = None 
  
  return cabam_data

def GenerateAssortativityDict(p_in):
    return {0: p_in, 1: 1-p_in }

def GenerateCABAMGraphWithFeatures(
    n, m, p_in, pi,
    feature_center_distance=0.0,
    feature_dim=0,
    num_feature_groups=1,
    feature_group_match_type=MatchType.RANDOM,
    feature_cluster_variance=1.0,
    normalize_features=True):

  result = CABAM()
  G, node_labels = GenerateCABAMGraph(n=n, m=m, c=num_feature_groups, native_probs=pi.tolist(), c_probs=GenerateAssortativityDict(p_in) )
  NetworkxToGraphWorldData(G, node_labels, result)

# borrowing node feature generation from SBM
  SimulateFeatures(result, feature_center_distance,
                   feature_dim,
                   num_feature_groups,
                   feature_group_match_type,
                   feature_cluster_variance,
                   normalize_features)

  return result

def GenerateCABAMGraph(n, m, c=2, native_probs=[0.5, 0.5], c_probs={1: 0.5, 0: 0.5}, logger=None):
    '''
    Main function for CABAM graph generation. Taken directly from https://github.com/nshah171/cabam-graph-generation/blob/master/cabam_utils.py
    
    n: maximum number of nodes
    m: number of edges to add at each timestep (also the minimum degree)
    c: number of classes
    native_probs: c-length vector of native class probabilities (must sum to 1)
    c_probs: p_c from the paper.  Entry for 1 (0) is the intra-class (inter-class) link strength.  Entries must sum to 1.
    
    Supports 3 variants of c_probs:
    -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
    -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
    -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
    '''
    
    if m < 1 or n < m:
        raise nx.NetworkXError(
                "NetworkXError must have m>1 and m<n, m=%d,n=%d" % (m, n))

    # graph initialization
    G = nx.empty_graph(m)
    intra_class_edges = 0
    inter_class_edges = 0
    total_intra_class = 0
    total_inter_class = 0
    
    intra_class_ratio_tracker = []
    alpha_tracker = []

    class_tbl = list(range(c))
    node_labels = np.array([np.random.choice(class_tbl, p=native_probs) for x in range(G.number_of_nodes())])
    node_degrees = np.array([1] * m) # technically degree 0, but using 1 here to make the math work out.

    # start adding nodes
    source = m
    source_label = np.random.choice(class_tbl, p=native_probs)
    # pbar = tqdm(total=n)
    # pbar.update(m)
    empirical_edge_fraction_to_degree_k = np.zeros(10)
    n_added = 0
    
    while source < n:
        if type(c_probs) == dict:
            if len(c_probs) == 2:
                # no funny business, just constants
                node_class_probs = np.array([c_probs[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            else:
                # pre-generated custom probabilities
                node_class_probs = np.array([c_probs[node_degrees[i]][abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
        else:
            # callable (function) probs
            node_class_probs = np.array([c_probs(node_degrees[i])[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            
        
        # determine m target nodes to connect to
        targets = []
        while len(targets) != m: 
            node_class_degree_probs = node_class_probs * node_degrees
            candidate_targets = np.where(node_class_degree_probs > 0)[0]

            if len(candidate_targets) >= m:
                # if we have enough qualifying nodes, sample from assortativity-weighted PA probs
                candidate_node_class_degree_probs = node_class_degree_probs[candidate_targets]
                candidate_node_class_degree_probs = candidate_node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
                targets = np.random.choice(candidate_targets, 
                                          size=m, 
                                          p=candidate_node_class_degree_probs, 
                                          replace=False)
            else:
                # else, use as many qualifying nodes as possible, and just sample from the PA probs for the rest.
                n_remaining_targets = m - len(candidate_targets)
                other_choices = np.where(node_class_degree_probs == 0)[0]
                other_node_degree_probs = node_degrees[other_choices]
                other_node_degree_probs = other_node_degree_probs / np.linalg.norm(other_node_degree_probs, ord=1)
                other_targets = np.random.choice(other_choices, 
                                                size=n_remaining_targets, 
                                                p=other_node_degree_probs,
                                                replace=False)
                #print(candidate_targets, candidate_targets.shape, other_targets, other_targets.shape)
                targets = np.concatenate((candidate_targets, other_targets))
            assert len(targets) == m

        G.add_edges_from([(source, target) for target in targets])
        edge_types = np.array([source_label == node_labels[target] for target in targets])
        intra_class_edges += np.count_nonzero(edge_types) # intra-class edges
        inter_class_edges += np.count_nonzero(edge_types == 0) # inter-class edges
        
        total_intra_class += np.count_nonzero(edge_types)
        total_inter_class +=  np.count_nonzero(edge_types == 0)
        total_intra_frac = total_intra_class / (total_intra_class + total_inter_class)
        
        #intra_class_ratio_tracker.append(total_intra_frac)
        #alpha_tracker.append(test_degree_distribution(node_degrees)[0])
        
        ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
        empirical_edge_fraction_to_degree_k += np.array([m*np.sum(ncdp[node_degrees == k]) for k in range(m, m+10)])
        
        if source % 500 == 0:
            theoretical_edge_fraction_to_degree_k = [((m*(m+1))/((k+1)*(k+2))) for k in range(m, m+10)]
            ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
            avgd_empirical_edge_fraction_to_degree_k = empirical_edge_fraction_to_degree_k / n_added
            
            # logger.info('Theor. edge prob to deg k: {}'.format(np.round(theoretical_edge_fraction_to_degree_k, 3)))
            # logger.info('Empir. edge prob to deg k: {}'.format(np.round(avgd_empirical_edge_fraction_to_degree_k, 3)))
            # snapshot_intra_frac = intra_class_edges / (intra_class_edges + inter_class_edges)
            # logger.info('Snapshot: ({}/{})={:.3f}\t Overall: {:.3f}'.format(intra_class_edges, intra_class_edges+inter_class_edges,
            #                                                                 snapshot_intra_frac, total_intra_frac))
            intra_class_edges = 0
            inter_class_edges = 0
            # logger.info('Max node degree: {}'.format(max(node_degrees)))

        # book-keeping
        node_degrees[targets] += 1
        node_labels = np.append(node_labels, source_label)
        node_degrees = np.append(node_degrees, m)
        # pbar.update(1)

        # move onto next node!
        n_added += 1
        source += 1
        source_label = np.random.choice(class_tbl, p=native_probs)
    
    # pbar.close()
    return G, node_labels