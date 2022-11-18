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

import graph_tool
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges


@dataclasses.dataclass
class LinkPredictionDataset:
  """Stores data for link prediction tasks.
  Attributes:
    graph: graph-tool Graph object.
    node_features: numpy array of node features.
    feature_memberships: list of integer node feature classes.
    graph_memberships: list of integer node classes. This is optional for many
      link prediction tasks, but if the generator for the task is some cluster
      model (such as the SBM), it may be useful to store class information for
      metric computation purposes.
  """
  graph: graph_tool.Graph = Ellipsis
  node_features: np.ndarray = Ellipsis
  edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis
  graph_memberships: np.ndarray = Ellipsis


def linkprediction_data_to_torchgeo_data(
    linkprediction_data: LinkPredictionDataset,
    training_ratio, tuning_ratio) -> Data:
  edge_tuples = []
  edge_feature_data = []
  for edge in linkprediction_data.graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])
    ordered_tuple = (edge[0], edge[1])
    if edge[0] > edge[1]:
      ordered_tuple = (edge[1], edge[0])
    edge_feature_data.append(
        linkprediction_data.edge_features[ordered_tuple])
    edge_feature_data.append(
        linkprediction_data.edge_features[ordered_tuple])

  node_features = torch.tensor(linkprediction_data.node_features,
                               dtype=torch.float)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  edge_attr = torch.tensor(edge_feature_data, dtype=torch.float)
  labels = torch.tensor(linkprediction_data.graph_memberships,
                        dtype=torch.long)
  torch_data = Data(x=node_features, edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_attr, y=labels)
  return train_test_split_edges(torch_data, val_ratio=tuning_ratio,
                                test_ratio=1.0 - training_ratio - tuning_ratio)
