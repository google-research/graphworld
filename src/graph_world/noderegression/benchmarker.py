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

from ..models.models import PyGBasicGraphModel
from ..beam.benchmarker import Benchmarker, BenchmarkerWrapper


class NodeRegressionBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']

    self._model = model_class(**h_params)
    # TODO(palowitch): make optimizer configurable.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=5e-4)
    self._criterion = torch.nn.MSELoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def SetMasks(self, train_mask, val_mask, test_mask):
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = test_mask

  def train_step(self, data):
    self._model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    out = self._model(data.x, data.edge_index).ravel()  # Perform a single forward pass.
    loss = self._criterion(out[self._train_mask],
                           data.y[self._train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    out = self._model(data.x, data.edge_index).ravel()
    if test_on_val:
      pred = out[self._val_mask].detach().numpy()
    else:
      pred = out[self._test_mask].detach().numpy()

    if test_on_val:
      correct = data.y[self._val_mask].numpy()
    else:
      correct = data.y[self._test_mask].numpy()

    results = {
        'mse': float(sklearn.metrics.mean_squared_error(correct, pred)),
    }
    return results

  def train(self, data,
            tuning_metric: str,
            tuning_metric_is_loss: bool):
    losses = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    best_val_metrics = None
    test_metrics = None
    for i in range(self._epochs):
      losses.append(float(self.train_step(data)))
      val_metrics = self.test(data, test_on_val=True)
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
    return losses, test_metrics, best_val_metrics


  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {
        'mse': -1,
    }
    out['test_metrics'] = {
        'mse': -1,
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    train_mask, val_mask, test_mask = masks

    self.SetMasks(train_mask, val_mask, test_mask)

    out['losses'] = None
    try:
      losses, test_metrics, val_metrics = self.train(
        torch_data, tuning_metric=tuning_metric,
        tuning_metric_is_loss=tuning_metric_is_loss)
      out['losses'] = losses
      out['val_metrics'].update(val_metrics)
      out['test_metrics'].update(test_metrics)
    except Exception as e:
      logging.info(f'Failed to run for sample id {sample_id}')
      out['skipped'] = True

    return out


@gin.configurable
class NodeRegressionBenchmark(BenchmarkerWrapper):
  def GetBenchmarker(self):
    return NodeRegressionBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NodeRegressionBenchmarker
