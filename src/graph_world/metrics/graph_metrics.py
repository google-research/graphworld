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
from typing import Dict

import graph_tool
from .graph_metrics_nx import graph_metrics_nx
import networkx as nx


def graph_metrics(graph: graph_tool.Graph) -> Dict[str, float]:
  """Computes graph metrics on a graph_tool graph object.

  Arguments:
    graph: graph_tool graph.
  Returns:
    dict from metric names to metric values.
  """
  nx_graph = nx.Graph()
  edge_list = [(int(e.source()), int(e.target())) for e in graph.edges()]
  nx_graph.add_edges_from(edge_list)
  return graph_metrics_nx(nx_graph)
