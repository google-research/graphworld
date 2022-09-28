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
import numpy as np

import graph_tool


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


def _get_edge_count_matrix(adj, labels):
  k = len(set(labels))
  seen_edges = set()
  edge_counts = np.zeros((k, k))
  edge_matrix = np.array(np.nonzero(adj))
  for edge_index in range(edge_matrix.shape[1]):
    v1 = edge_matrix[0, edge_index]
    v2 = edge_matrix[1, edge_index]
    edge_tuple = (v1, v2)
    if edge_tuple in seen_edges:
      continue
    else:
      seen_edges.add(edge_tuple)
    k1 = labels[v1]
    k2 = labels[v2]
    edge_counts[k1, k2] += 1
    if k1 == k2:
      edge_counts[k1, k2] += 1
  edge_counts = edge_counts.astype(np.int32)
  return edge_counts

def matrix_row_norm(X):
  return X / np.linalg.norm(X, axis=1)[:, None]

def _get_degrees_by_labels(labels, degrees, adjusted=False):
  if adjusted:
    degrees_by_labels = {
        label: np.sum(degrees[np.where(labels == label)[0]]) for
        label in labels}
  else:
    degrees_by_labels = {
        label: len(np.where(labels == label)[0]) for
        label in labels}
  return degrees_by_labels


def _get_p_to_q_ratio(G, labels, degrees, adjusted=False):
  adj = graph_tool.spectral.adjacency(G)
  edge_count_matrix = _get_edge_count_matrix(adj, labels)
  pi = _get_pi(labels, degrees, adjusted)
  n = adj.shape[0]
  num_within_pairs = np.sum(pi ** 2.0) * (n ** 2.0)
  num_between_pairs = (n ** 2.0) - num_within_pairs
  num_within_edges = np.sum(np.diag(edge_count_matrix))
  num_between_edges = np.sum(edge_count_matrix) - num_within_edges
  return ((num_within_edges / num_within_pairs) /
          (num_between_edges / num_between_pairs))


def _get_pareto_exponent(degrees):
  n = len(degrees)
  dmin = np.min(degrees)
  alpha = n / np.sum(np.log(degrees / dmin))
  return alpha


def _get_pi(labels, degrees=None, adjusted=False):
  label_counter = _get_degrees_by_labels(labels, degrees, adjusted)
  sizes = np.array(sorted([v for v in label_counter.values()]))
  return sizes / np.sum(sizes)


def _get_community_size_simpsons(labels):
  pi = _get_pi(labels)
  return np.sum(pi ** 2.0)


def _get_num_clusters(labels):
  pi = _get_pi(labels)
  return len(pi)


def _get_average_degree(degrees):
  return np.mean(degrees)


def NodeLabelMetrics(graph, labels, features):
  metrics = {'edge_homogeneity': edge_homogeneity(graph, labels)}
  normed_features = matrix_row_norm(features)
  in_avg, out_avg = feature_homogeneity(normed_features, labels)
  metrics.update({'avg_in_feature_angular_distance': in_avg,
                  'avg_out_feature_angular_distance': out_avg,
                  'feature_angular_snr': in_avg/out_avg})
  degrees = graph.get_out_degrees(graph.get_vertices())
  nonzero_degrees = np.array([d for d in degrees if d > 0])
  metrics['pareto_exponent'] = _get_pareto_exponent(nonzero_degrees)
  metrics['avg_degree_est'] = _get_average_degree(degrees)
  if labels is not None:
    metrics['community_size_simpsons'] = _get_community_size_simpsons(labels)
    metrics['p_to_q_ratio_est'] = _get_p_to_q_ratio(graph, labels, degrees)
    metrics['p_to_q_ratio__est_dc'] = _get_p_to_q_ratio(graph, labels, degrees,
                                                  adjusted=True)
    metrics['num_clusters'] = _get_num_clusters(labels)
  return metrics