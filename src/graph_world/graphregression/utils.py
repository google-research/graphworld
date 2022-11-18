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

import math
import random

import dataclasses
from typing import Dict, List, Optional, Tuple
import graph_tool
import numpy as np
import torch
from torch_geometric.data import Data


@dataclasses.dataclass
class GraphRegressionDataset:
  """Stores data for graph regression tasks.
  Attributes:
    graphs: a list of graph-tool Graph objects
    graph_node_features: a list numpy node feature matrices, one for each graph
    graph_regression_target: float-convertible np vector of graph values.
  """
  graphs: List[graph_tool.Graph] = Ellipsis
  graph_node_features: List[np.ndarray] = Ellipsis
  graph_regression_target: np.ndarray = Ellipsis


def graph_regression_dataset_example_to_torch_geo_data(
    graph: graph_tool.Graph, target: float,
    features: Optional[np.ndarray] = None) -> Data:
  edge_tuples = []
  for edge in graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])

  node_features = torch.from_numpy(features)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  return Data(x=node_features, edge_index=edge_index.t().contiguous(),
              y=float(target))

