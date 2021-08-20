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

# @title Internal code for SBM generator class
# pasted from https://github.com/google-research/google-research/blob/master/graph_embedding/simulations/sbm_simulator.py
import collections
import enum
import math
import random
from typing import Dict, Sequence, List, Tuple

import dataclasses
import gin
import graph_tool
import numpy as np

from substructure.utils import erdos_graph


def _get_star_graph():
  g = graph_tool.Graph(directed=False)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(0, 3)
  return g


def _get_triangle_graph():
  g = graph_tool.Graph(directed=False)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  g.add_edge(2, 0)
  return g


def _get_tailed_triangle_graph():
  g = _get_triangle_graph()
  g.add_edge(0, 3)
  return g


def _get_chordal_cycle_graph():
  g = _get_tailed_triangle_graph()
  g.add_edge(1, 3)
  return g


@gin.constants_from_enum
class Substructure(enum.Enum):
  """Indicates type of substructure to count.
  """
  STAR_GRAPH = 1
  TRIANGLE_GRAPH = 2
  TAILED_TRIANGLE_GRAPH = 3
  CHORDAL_CYCLE_GRAPH = 4


def GetSubstructureGraph(substruct: Substructure):
  if substruct == Substructure.STAR_GRAPH:
    return _get_star_graph()
  elif substruct == Substructure.TRIANGLE_GRAPH:
    return _get_triangle_graph()
  elif substruct == Substructure.CHORDAL_CYCLE_GRAPH:
    return _get_chordal_cycle_graph()
  elif substruct == Substructure.TAILED_TRIANGLE_GRAPH:
    return _get_tailed_triangle_graph()


def GenerateSubstructureDataset(
    num_graphs: int,
    num_vertices: int,
    edge_prob: float,
    substruct_graph: graph_tool.Graph):
  graphs = [erdos_graph(num_vertices, edge_prob) for _ in range(num_graphs)]
  substruct_counts = []
  for graph in graphs:
    assert graph.num_vertices() == num_vertices, "num_vertices is %d" % graph.num_vertices()
    _, counts = graph_tool.clustering.motifs(
      g=graph,
      k=substruct_graph.num_vertices(),
      motif_list=[substruct_graph]
    )
    substruct_counts.append(counts[0])
  return {'graphs': graphs, 'substruct_counts': substruct_counts}
