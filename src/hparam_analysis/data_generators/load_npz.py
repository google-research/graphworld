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

import os.path as osp
import re
from typing import Callable, List, Optional

import graph_tool

from google.cloud import storage
import google.oauth2.credentials
import numpy as np
import scipy.sparse as sps
import torch
from torch_geometric.data import Data, InMemoryDataset


def _load_npz_to_sparse_graph(file_name):
  with np.load(file_name, allow_pickle=True) as loader:
    loader = dict(loader)
    adj_matrix = sps.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'],
    )

    if 'attr_data' in loader:
      # Attributes are stored as a sparse CSR matrix
      attr_matrix = sps.csr_matrix(
          (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
          shape=loader['attr_shape'],
      ).todense()
    elif 'attr_matrix' in loader:
      # Attributes are stored as a (dense) np.ndarray
      attr_matrix = loader['attr_matrix']
    else:
      raise Exception('No attributes in the data file', file_name)

    if 'labels_data' in loader:
      # Labels are stored as a CSR matrix
      labels = sps.csr_matrix(
          (
              loader['labels_data'],
              loader['labels_indices'],
              loader['labels_indptr'],
          ),
          shape=loader['labels_shape'],
      )
      label_mask = labels.nonzero()[0]
      labels = labels.nonzero()[1]
    elif 'labels' in loader:
      # Labels are stored as a numpy array
      labels = loader['labels']
      label_mask = np.ones(labels.shape, dtype=bool)
    else:
      raise Exception('No labels in the data file', file_name)

  return adj_matrix, attr_matrix, labels, label_mask

def _get_gt_graph(adj_matrix):
  gt_graph = graph_tool.Graph()
  gt_graph.add_edge_list(np.array(adj_matrix.nonzero()).T)
  # Ensure that all edges reciprocate
  for e in list(gt_graph.edges()):
    gt_graph.add_edge(e.target(), e.source())
  return gt_graph


class NpzDataset(InMemoryDataset):
  r"""Modified torch_geometric.datasets.Planetoid for NPZ file reading for GCP.

  Args:
      url (string): Root of datasets folder containing named dataset subfolders.
      root (string): Root directory where the dataset should be saved.
      name (string): The name of the dataset.
      transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
  """

  def __init__(
      self,
      url: str,
      root: str,
      name: str,
      project_name: str,
      access_token: None,
      transform: Optional[Callable] = None,
      pre_transform: Optional[Callable] = None,
  ):
    if access_token is None:
      self.gcs_client = storage.Client(project_name)
    else:
      self.gcs_client = storage.Client(
          project_name,
          credentials=google.oauth2.credentials.Credentials(access_token),
      )
    self.gcs_path_regex = re.compile(r'gs:\/\/([^\/]+)\/(.*)')
    self.name = name
    self.gt_data = None
    self.url = url
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])
    self.data.y = self.data.y.to(torch.long)
    self._get_gt_data()

  def _get_gt_data(self):
    with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as infile:
      adj_matrix, attr_matrix, labels, label_mask = _load_npz_to_sparse_graph(
          infile
      )
    self.gt_data = {'gt_graph': _get_gt_graph(adj_matrix),
                    'labels': labels,
                    'features': attr_matrix}

  @property
  def raw_dir(self) -> str:
    return self.root

  @property
  def processed_dir(self) -> str:
    return osp.join(self.root, 'processed')

  @property
  def raw_file_names(self) -> List[str]:
    return [f'{self.name}.npz']

  @property
  def processed_file_names(self) -> str:
    return f'{self.name}.pt'

  def download(self):
    for name in self.raw_file_names:
      gcs_path = f'{self.url}/{self.name.lower()}.npz'
      file_match = self.gcs_path_regex.match(gcs_path)
      bucket = self.gcs_client.get_bucket(file_match.group(1))
      blob = bucket.get_blob(file_match.group(2))
      blob.download_to_filename(f'{self.raw_dir}/{name}')

  def process(self):
    with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as infile:
      adj_matrix, attr_matrix, labels, label_mask = _load_npz_to_sparse_graph(
          infile
      )
    data = Data(
        x=torch.tensor(attr_matrix),
        edge_index=torch.tensor(np.vstack(adj_matrix.nonzero()),
                                dtype=torch.long),
        y=torch.tensor(labels),
    )
    if label_mask.dtype == bool and label_mask.sum() == label_mask.shape[0]:
                data.label_mask = torch.arange(label_mask.shape[0],
                                               dtype=torch.long)
    else:
                data.label_mask = torch.LongTensor(label_mask)
    data = data if self.pre_transform is None else self.pre_transform(data)
    torch.save(self.collate([data]), self.processed_paths[0])

  def __repr__(self) -> str:
    return f'{self.name}()'
