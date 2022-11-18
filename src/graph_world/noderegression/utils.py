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
import dataclasses
from typing import Dict, Tuple

import numpy as np
import graph_tool
from graph_tool.all import *
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import Data

@dataclasses.dataclass
class NodeRegressionDataset:
  """Stores data for node regression tasks.
  Attributes:
    graph: graph-tool Graph object.
    node_regression_target: numpy array of float-castable regression targets.
    node_features: numpy array of node features.
    edge_features: map from edge tuple to numpy array. Only stores undirected
      edges, i.e. (0, 1) will be in the map, but (1, 0) will not be.
    graph_memberships: list of integer node classes. This is optional for many
      node regression tasks, but if the generator for the task is some cluster
      model (such as the SBM), it may be useful to store class information for
      metric computation purposes.
  """
  graph: graph_tool.Graph = Ellipsis
  node_regression_target: np.ndarray = Ellipsis
  node_features: np.ndarray = Ellipsis
  edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis
  graph_memberships: np.ndarray = Ellipsis


def sample_masks(n_elements: int, n_train: float, n_tune: float):
  indices = np.arange(n_elements, dtype=int)
  train_indices = np.random.choice(indices, int(n_train * n_elements), replace=False)
  remainder = np.setdiff1d(indices, train_indices)
  val_indices = np.random.choice(remainder, int(n_tune * n_elements), replace=False)
  test_indices = np.setdiff1d(remainder, val_indices)
  return train_indices, val_indices, test_indices


def calculate_target(graph: graph_tool.Graph, target: str) -> np.ndarray:
  if target == 'pagerank':
    return np.fromiter(graph_tool.centrality.pagerank(graph).a, float)
  if target == 'betweenness':
    return np.fromiter(graph_tool.centrality.betweenness(graph)[0].a, float)
  if target == 'closeness':
    return np.fromiter(graph_tool.centrality.closeness(graph).a, float)
  if target == 'eigenvector':
    return np.fromiter(graph_tool.centrality.eigenvector(graph)[1].a, float)
  if target == 'katz':
    return np.fromiter(graph_tool.centrality.katz(graph).a, float)
  if target == 'hits_authority':
    return np.fromiter(graph_tool.centrality.hits(graph)[1].a, float)
  if target == 'hits_hub':
    return np.fromiter(graph_tool.centrality.hits(graph)[2].a, float)
  if target == 'local_clustering':
    return np.fromiter(graph_tool.clustering.local_clustering(graph).a, float)
  if target == 'kcore':
    return np.fromiter(graph_tool.topology.kcore_decomposition(graph).a, float)
  raise ValueError('Unknown target! Received', target)


def noderegression_data_to_torchgeo_data(
    noderegression_data: NodeRegressionDataset) -> Data:
  edge_tuples = []
  edge_feature_data = []
  for edge in noderegression_data.graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])
    ordered_tuple = (edge[0], edge[1])
    if edge[0] > edge[1]:
      ordered_tuple = (edge[1], edge[0])
    edge_feature_data.append(
        noderegression_data.edge_features[ordered_tuple])
    edge_feature_data.append(
        noderegression_data.edge_features[ordered_tuple])

  node_features = torch.tensor(noderegression_data.node_features,
                               dtype=torch.float)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  return Data(x=node_features, edge_index=edge_index.t().contiguous(),
              y=torch.tensor(
                  noderegression_data.node_regression_target,
                  dtype=torch.float))
