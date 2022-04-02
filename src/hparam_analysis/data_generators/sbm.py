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

import re

import graph_tool
import numpy as np

from google.cloud import storage
from graph_tool.all import *
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

def _get_edge_count_matrix(data):
  labels = data.y.numpy()
  k = len(set(labels))
  seen_edges = set()
  edge_counts = np.zeros((k, k))
  edge_matrix = data.edge_index.numpy()
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


def get_adj_from_file(sim_adj_file):
  gcs_client = storage.Client()
  gcs_path_regex = re.compile(r'gs:\/\/([^\/]+)\/(.*)')
  file_match = gcs_path_regex.match(sim_adj_file)
  bucket = gcs_client.get_bucket(file_match.group(1))
  blob = bucket.get_blob(file_match.group(2))
  blob.download_to_filename('/tmp/adj')
  adj = np.loadtxt('/tmp/adj')
  return adj

def get_sbm_from_torchgeo_data(data, sim_adj_file=None):
  adj = np.squeeze(to_dense_adj(data['edge_index']).numpy())
  labels = data.y.numpy()
  k = len(set(labels))
  degrees = list(np.sum(adj, axis=1))
  sbm = graph_tool.generation.generate_sbm(
    b=list(data.y.numpy()),
    probs=_get_edge_count_matrix(data) / 2.0,
    out_degs=degrees
  )

  edge_set = set()

  # If sim_adj_file is provided, load sbm adj from disk and convert.
  if sim_adj_file is not None:
    sbm_adj = get_adj_from_file(sim_adj_file)
    nonzero_a, nonzero_b = np.nonzero(sbm_adj)
    for e_tuple in zip(nonzero_a, nonzero_b):
      a = int(e_tuple[0])
      b = int(e_tuple[1])
      if (a, b) in edge_set:
        continue
      edge_set.add((a, b))
      edge_set.add((b, a))
  else:
    for e in sbm.edges():
      a = int(e.source())
      b = int(e.target())
      if (a, b) in edge_set:
        continue
      edge_set.add((a, b))
      edge_set.add((b, a))

  edge_index1 = [0] * len(edge_set)
  edge_index2 = [0] * len(edge_set)
  sbm_edge_counts = np.zeros((k, k))
  for i, e in enumerate(edge_set):
    edge_index1[i] = e[0]
    edge_index2[i] = e[1]
    label0 = labels[e[0]]
    label1 = labels[e[1]]
    sbm_edge_counts[label0, label1] += 1
    sbm_edge_counts[label1, label0] += 1

  edge_index = torch.tensor([edge_index1,
                             edge_index2], dtype=torch.long)
  sbm_edge_counts = sbm_edge_counts.astype(np.int32)
  sbm_data = Data(x=data.x, edge_index=edge_index, y=data.y)
  sbm_data.train_mask = data.train_mask.clone()
  sbm_data.val_mask = data.val_mask.clone()
  sbm_data.test_mask = data.test_mask.clone()
  return sbm_data, sbm_edge_counts