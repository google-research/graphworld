import numpy as np
import graph_tool
from graph_tool.all import *

from ..sbm.sbm_simulator import StochasticBlockModel

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