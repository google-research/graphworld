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

import os

import apache_beam as beam
from ..data_generators.load_npz import NpzDataset
from ..data_generators.sbm import get_sbm_from_torchgeo_data
import gin
import numpy as np
import torch
from ..utils.test_gcn import test_gcn
from graph_world.metrics.graph_metrics import graph_metrics
from graph_world.metrics.node_label_metrics import NodeLabelMetrics

from graph_world.nodeclassification.utils import get_label_masks


class GcnTester(beam.DoFn):

  def __init__(self, random_seeds, sim=False, dataset_path='', dataset_name='',
               sim_adj_file=None, num_train_per_class=20, num_val=500,
               root='/tmp', project_name="", gcs_auth=None):
    self._random_seeds = random_seeds
    self._dataset_path = dataset_path
    self._dataset_name = dataset_name
    self._root = root
    self._sim = sim
    self._sim_adj_file = sim_adj_file
    self._num_train_per_class = num_train_per_class
    self._project_name = project_name
    self._gcs_auth = gcs_auth
    self._num_val = num_val

  def process(self, test_config):
    output = test_config

    # Load dataset
    dataset = NpzDataset(url=self._dataset_path,
                         root=os.path.join(self._root, self._dataset_name),
                         name=self._dataset_name,
                         project_name=self._project_name,
                         access_token=self._gcs_auth)
    data = dataset[0]

    # Row-normalize features
    X = np.squeeze(data.x.numpy())
    data.x = torch.tensor((X.T / np.sum(X, axis=1)).T)

    # Swap to simulated data if desired
    if self._sim:
      data, _ = get_sbm_from_torchgeo_data(
          data, os.path.join(self._dataset_path, self._sim_adj_file))

    # Get metrics for all seeds
    val_accs = []
    test_accs = []
    epochs = []
    for random_seed in self._random_seeds:
      data.train_mask, data.val_mask, data.test_mask = get_label_masks(
          data.y, self._num_train_per_class, self._num_val, random_seed)

      best_val_acc, best_test_acc, epoch_count = test_gcn(
          data,
          hidden_channels=test_config['hidden_channels'],
          weight_decay=test_config['weight_decay'],
          lr=test_config['learning_rate'],
          dropout=test_config['dropout'],
          num_layers=test_config['num_layers']
      )

      val_accs.append(best_val_acc)
      test_accs.append(best_test_acc)
      epochs.append(epoch_count)

    output['val_acc_mean'] = np.mean(val_accs)
    output['test_acc_mean'] = np.mean(test_accs)
    output['epoch_mean'] = np.mean(epochs)
    output['val_acc_std'] = np.std(val_accs)
    output['test_acc_std'] = np.std(test_accs)
    output['epoch_std'] = np.std(epochs)

    # Compute graph metrics
    if test_config['index'] == 0:
      output.update(graph_metrics(dataset.gt_data['gt_graph']))
      output.update(NodeLabelMetrics(dataset.gt_data['gt_graph'],
                                     dataset.gt_data['labels'],
                                     dataset.gt_data['features']))
    yield output


@gin.configurable
class HparamBeamHandler:
  """Wrapper for hparam pipeline

  Arguments:
    sim_adj_file: name of numpy (dense) adjacency matrix to swap in for sbm.
      filename must be relative to 'dataset_path'.

  """

  def __init__(self, random_seeds, sim=False, dataset_path='', dataset_name='',
               project_name='', gcs_auth=None, sim_adj_file=None):
    self._random_seeds = random_seeds
    self._dataset_path = dataset_path
    self._dataset_name = dataset_name
    self._sim = sim
    self._sim_adj_file = sim_adj_file
    self._project_name = project_name
    self._gcs_auth = gcs_auth

  def GetGcnTester(self):
    return GcnTester(self._random_seeds, self._sim, self._dataset_path,
                     self._dataset_name, self._sim_adj_file,
                     project_name=self._project_name, gcs_auth=self._gcs_auth)
