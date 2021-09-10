#@title Imports
import os
import numpy as np
import graph_tool


def gini(array):
    array = array.astype(np.float32)
    array += np.finfo(np.float32).eps
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array))


def diameter_lb(graph, n_tries=100):
  max_diameter = -1
  nodes = np.random.choice(list(graph.vertices()), n_tries)
  for node in nodes:
    diameter = graph_tool.topology.pseudo_diameter(graph, node)[0]
    if diameter > max_diameter:
      max_diameter = diameter
  return max_diameter


def bfs_sp(graph, source, others):
  others = set(others)
  for edge in graph_tool.search.bfs_iterator(graph, source):
    if edge.target() in others:
      return graph_tool.topology.shortest_distance(graph, source, edge.target())
  return -1


def average_cc_sp_length(G_comp):
  sp_lengths = 0
  sp_distances = graph_tool.topology.shortest_distance(G_comp, directed=False)
  for vtx in G_comp.vertices():
    sp_lengths += np.sum(sp_distances[vtx].a)
  return sp_lengths / G_comp.num_vertices() / (G_comp.num_vertices() - 1)


def GraphMetrics(G):
  result = {}
  result['n_nodes'] = G.num_vertices()
  result['n_edges'] = G.num_edges()
  result['edge_density'] = G.num_edges() / G.num_vertices() / (G.num_vertices() - 1)
  result['avg_in_degree'] = np.mean(G.get_in_degrees(G.get_vertices()))
  result['edge_reciprocity'] = graph_tool.topology.edge_reciprocity(G)
  result['avg_undirected_degree'] = np.mean(G.get_out_degrees(G.get_vertices())) / 2
  result['degree_gini'] = gini(G.get_out_degrees(G.get_vertices()))
  result['pseudo_diameter'] = diameter_lb(G)
  coreness = np.array(graph_tool.topology.kcore_decomposition(G).a)
  result['coreness_eq_1'] = np.sum(coreness == 1) / G.num_vertices()
  result['coreness_geq_2'] = np.sum(coreness >= 2) / G.num_vertices()
  result['coreness_geq_5'] = np.sum(coreness >= 5) / G.num_vertices()
  result['coreness_geq_10'] = np.sum(coreness >= 10) / G.num_vertices()
  result['coreness_gini'] = gini(coreness)
  result['avg_local_cc'] = np.mean(np.array(graph_tool.clustering.local_clustering(G).a))
  result['global_cc'] = graph_tool.clustering.global_clustering(G)[0]
  comp = graph_tool.topology.label_largest_component(G)
  result['cc_size'] = int(comp.a.sum()) / G.num_vertices()
  return result
