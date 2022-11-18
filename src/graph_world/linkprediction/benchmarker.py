# Copyright 2022 Google LLC
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
"""
There are currently 3 pieces required for each model:

  * BenchmarkerWrapper (ex. NodeGCN) -- Used in GIN config, this delegates to the Benchmarker.
  * ModelBenchmarker (ex. GCNNodeBenchmarker) -- This performs the actual training and eval steps for the model
  * Modelmpl (ex. GCNNodeModel) -- This is the actual model implemention (wrapping together convolution layers)
"""
import copy
import gin
import logging
import numpy as np
import graph_tool.all as gt
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import torch
from torch_geometric.nn import GAE

from ..models.models import PyGBasicGraphModel
from ..beam.benchmarker import Benchmarker, BenchmarkerWrapper


# Link prediction
class LPBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params):

    super().__init__(generator_config, model_class, benchmark_params, h_params)

    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']
    self._model = self._model_class(**h_params)
    self._lp_wrapper_model = GAE(self._model)
    # TODO(palowitch,tsitsulin): fill optimizer using param input instead.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=5e-4)

  def AdjustParams(self, generator_config):
    if 'num_clusters' in generator_config:
      self._h_params['out_channels'] = generator_config['num_clusters']

  def train_step(self, data):
    self._model.train()
    self._lp_wrapper_model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    z = self._model(data.x, data.train_pos_edge_index)
    loss = self._lp_wrapper_model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    self._lp_wrapper_model.eval()
    results = {}
    z = self._model(data.x, data.train_pos_edge_index)

    if test_on_val:
      roc_auc_score, average_precision_score = self._lp_wrapper_model.test(
        z, data.val_pos_edge_index, data.val_neg_edge_index)
    else:
      roc_auc_score, average_precision_score = self._lp_wrapper_model.test(
        z, data.test_pos_edge_index, data.test_neg_edge_index)
    results['rocauc'] = roc_auc_score
    results['ap'] = average_precision_score
    return results

  def train(self, data):
    losses = []
    for epoch in range(self._epochs):
      losses.append(float(self.train_step(data)))
    return losses

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }
    out['test_metrics'] = {
        'rocaus': 0,
        'ap': 0,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    out['losses'] = None
    try:
      out['losses'] = self.train(torch_data)
      # Divide by zero sometimes happens with the ksample masks.
      out['val_metrics'].update(self.test(torch_data, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, test_on_val=False))
    except Exception:
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    return out


class LPBaselineBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    self._scorer = h_params['scorer']

  def score(self, graph, edge_index):
    return gt.vertex_similarity(graph, sim_type=self._scorer,
                                vertex_pairs=edge_index)

  def test(self, data, test_on_val=False):
    graph = gt.Graph(directed=False)
    graph.add_vertex(data.y.shape[0])
    graph.add_edge_list(data.train_pos_edge_index.T)

    if test_on_val:
      pos_scores = self.score(graph, data.val_pos_edge_index.T)
      neg_scores = self.score(graph, data.val_neg_edge_index.T)
      y_true = np.ones(data.val_pos_edge_index.shape[1] +
                       data.val_neg_edge_index.shape[1])
      y_true[data.val_pos_edge_index.shape[1]:] = 0
    else:
      pos_scores = self.score(graph, data.test_pos_edge_index.T)
      neg_scores = self.score(graph, data.test_neg_edge_index.T)
      y_true = np.ones(data.test_pos_edge_index.shape[1] +
                       data.test_neg_edge_index.shape[1])
      y_true[data.test_pos_edge_index.shape[1]:] = 0

    all_scores = np.hstack([pos_scores, neg_scores])
    all_scores = np.nan_to_num(all_scores, copy=False)

    return {
        'rocauc': sklearn.metrics.roc_auc_score(y_true, all_scores),
        'ap': sklearn.metrics.average_precision_score(y_true, all_scores),
    }

  def GetModelName(self):
    return 'Baseline'

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }
    out['test_metrics'] = {
        'rocauc': 0,
        'ap': 0,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    try:
      # Divide by zero sometimes happens with the ksample masks.
      out['val_metrics'].update(self.test(torch_data, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, test_on_val=False))
    except Exception:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      out['skipped'] = True

    return out

@gin.configurable
class LPBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LPBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return LPBenchmarker

@gin.configurable
class LPBenchmarkBaseline(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LPBaselineBenchmarker(None, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return LPBaselineBenchmarker
