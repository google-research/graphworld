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

import math
import random

import graph_tool
import numpy as np
import torch
from torch_geometric.data import Data


def substructure_graph_to_torchgeo_data(
    substruct_graph: graph_tool.Graph, substruct_count: float) -> Data:
  edge_tuples = []
  for edge in substruct_graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])

  node_features = torch.ones([substruct_graph.num_vertices(), 1], dtype=torch.float)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  return Data(x=node_features, edge_index=edge_index.t().contiguous(),
              y=float(substruct_count))


def erdos_graph(num_vertices, edge_prob):
  num_edges = np.random.binomial
  g = graph_tool.Graph(directed=False)
  if edge_prob == 0.0:
    return graph_tool.Graph(directed=False)
  _ = g.add_vertex(num_vertices)
  for u in range(num_vertices - 1):
    v = u + 1
    while v < num_vertices:
      r = random.uniform(0.0, 1.0)
      v = v + int(math.floor(math.log(r) / math.log(1.0 - edge_prob)))
      if v < num_vertices:
        g.add_edge(u, v)
        v = v + 1
  return g
