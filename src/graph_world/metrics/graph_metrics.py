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


def matrix_row_norm(X):
  return X / np.linalg.norm(X, axis=1)[:, None]


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


def edge_homogeneity(graph, labels):
  count_in = 0
  count_total = 0
  for edge in graph.edges():
    count_total += 1
    count_in += (
        labels[int(edge.source())] == labels[int(edge.target())])
  return count_in / count_total


def sum_angular_distance_matrix_nan(X, Y, batch_size=100):
  nx = X.shape[0]
  ny = Y.shape[0]

  total_sum = 0.0

  pos1 = 0
  while pos1 < nx:
    end1 = min(pos1 + batch_size, nx)
    vec1 = X[pos1:end1, :]

    pos2 = 0
    curr_sum = 0.0
    while pos2 < ny:
      # Get data
      end2 = min(pos2 + batch_size, ny)
      vec2 = Y[pos2:end2, :]

      vecsims = np.matmul(vec1, vec2.T)
      vecsims = np.clip(vecsims, -1, 1)
      vecsims = 1.0 - np.arccos(vecsims) / np.pi
      vecsims[np.where(np.isnan(vecsims))] = 1.0
      total_sum += np.sum(vecsims)

      pos2 += batch_size

    pos1 += batch_size
  return total_sum


def feature_homogeneity(normed_features, labels):
  all_labels = sorted(list(set(labels)))
  n_labels = len(all_labels)
  sum_mat = np.zeros((n_labels, n_labels))
  count_mat = np.zeros((n_labels, n_labels))
  for label_idx, i in enumerate(all_labels):
    idx_i = np.where(labels == i)[0]
    vecs_i = normed_features[idx_i, :]
    for j in all_labels[label_idx:]:
      idx_j = np.where(labels == j)[0]
      vecs_j = normed_features[idx_j, :]
      the_sum = sum_angular_distance_matrix_nan(vecs_i, vecs_j)
      the_count = len(idx_j) * len(idx_i)
      if i == j:
        the_sum -= float(len(idx_j))
        the_sum /= 2.0
        the_count -= float(len(idx_j))
        the_count /= 2
      sum_mat[i, j] = the_sum
      count_mat[i, j] = the_count
  out_avg = np.sum(sum_mat[np.triu_indices(sum_mat.shape[0])]) / (
    np.sum(count_mat[np.triu_indices(count_mat.shape[0])]))
  in_avg = np.sum(np.diag(sum_mat)) / np.sum(np.diag(count_mat))
  return in_avg, out_avg


def NodeLabelMetrics(graph, labels, features):
  metrics = {'edge_homogeneity': edge_homogeneity(graph, labels)}
  normed_features = matrix_row_norm(features)
  in_avg, out_avg = feature_homogeneity(normed_features, labels)
  metrics.update({'avg_in_feature_angular_distance': in_avg,
                  'avg_out_feature_angular_distance': out_avg,
                  'feature_angular_snr': in_avg/out_avg})
  return metrics