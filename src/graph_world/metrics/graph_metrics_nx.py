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
import networkx as nx
import numpy as np


def _degrees(graph: nx.Graph) -> np.ndarray:
  """Returns degrees of the input graph."""
  return np.array([d for _, d in graph.degree()]).astype(np.float32)


def _counts(graph: nx.Graph) -> Dict[str, float]:
  """Returns a dict of count statistics on a graph.

  Arguments:
    graph: a networkx Graph object.
  Returns:
    dict with the following keys and values:
      num_nodes: count of nodes in graph
      num_edges: number of edges in graph
      edge_density: num_edges / {num_nodes choose 2}
  """
  num_nodes = float(graph.number_of_nodes())
  num_edges = float(graph.number_of_edges()) * 2.0  # count both directions
  edge_density = 0.0
  if num_nodes > 1.0:
    edge_density = num_edges / num_nodes / (num_nodes - 1.0)
  return {'num_nodes': num_nodes, 'num_edges': num_edges,
          'edge_density': edge_density}


def _gini_coefficient(array: np.ndarray) -> float:
  """Computes the Gini coefficient of a 1-D input array."""
  if array.size == 0:  # pylint: disable=g-explicit-length-test  (numpy arrays have no truth value)
    return 0.0
  array = array.astype(np.float32)
  array += np.finfo(np.float32).eps
  array = np.sort(array)
  n = array.shape[0]
  index = np.arange(1, n + 1)
  return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array))


def _diameter(graph: nx.Graph) -> float:
  """Computes diameter of the graph."""
  if graph.number_of_nodes() == 0:
    return 0.0
  if not nx.is_connected(graph):
    return np.inf
  return float(nx.diameter(graph))


def _largest_connected_component_size(graph: nx.Graph) -> float:
  """Computes the relative size of the largest graph connected component."""
  if graph.number_of_nodes() == 0:
    return 0.0
  if graph.number_of_nodes() == 1:
    return 1.0
  components = nx.connected_components(graph)
  return np.max(list(map(len, components))) / graph.number_of_nodes()


def _power_law_estimate(degrees: np.ndarray) -> float:
  degrees = degrees + 1.0
  n = degrees.shape[0]
  return 1.0 + n / np.sum(np.log(degrees / np.min(degrees)))


def graph_metrics_nx(graph: nx.Graph) -> Dict[str, float]:
  """Computes graph metrics on a networkx graph object.

  Arguments:
    graph: networkx graph.
  Returns:
    dict from metric names to metric values.
  """
  result = _counts(graph)
  degrees = _degrees(graph)
  result['degree_gini'] = _gini_coefficient(degrees)
  result['approximate_diameter'] = _diameter(graph)
  if graph.number_of_nodes() == 0:  # avoid np.mean of empty slice
    result['avg_degree'] = 0.0
    return result
  result['avg_degree'] = float(np.mean(degrees))
  core_numbers = np.array(list(nx.core_number(graph).values()))
  result['coreness_eq_1'] = float(np.mean(core_numbers == 1))
  result['coreness_geq_2'] = float(np.mean(core_numbers >= 2))
  result['coreness_geq_5'] = float(np.mean(core_numbers >= 5))
  result['coreness_geq_10'] = float(np.mean(core_numbers >= 10))
  result['coreness_gini'] = float(_gini_coefficient(core_numbers))
  result['avg_cc'] = float(np.mean(list(nx.clustering(graph).values())))
  result['transitivity'] = float(nx.transitivity(graph))
  result['num_triangles'] = float(
      np.sum(list(nx.triangles(graph).values())) / 3.0)
  result['cc_size'] = float(_largest_connected_component_size(graph))
  result['power_law_estimate'] = _power_law_estimate(degrees)
  return result
