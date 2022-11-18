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
import gin
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
import torch

from ..models.utils import MseWrapper
from ..models.models import PyGBasicGraphModel
from ..beam.benchmarker import Benchmarker, BenchmarkerWrapper


@gin.configurable
class NNGraphBenchmark(BenchmarkerWrapper):

  def __init__(self, model_class, benchmark_params, h_params):
    self._model_class = model_class
    self._benchmark_params = benchmark_params
    self._h_params = h_params

  def GetBenchmarker(self):
    return NNGraphBenchmarker(self._model_class, self._benchmark_params, self._h_params)

  def GetBenchmarkerClass(self):
    return NNGraphBenchmarker


class NNGraphBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params):
    super().__init__(generator_config, model_class, benchmark_params, h_params)
    self._epochs = self._benchmark_params['epochs']
    self._lr = self._benchmark_params['lr']
    self._model = PyGBasicGraphModel(self._model_class, self._h_params)
    # TODO(palowitch) make optimizer configurable
    self._optimizer = torch.optim.Adam(self._model.parameters(), self._lr, weight_decay=5e-4)
    self._criterion = torch.nn.MSELoss()

  def train(self, loader):
    self._model.train()
    train_losses = []
    train_mse = []
    for epoch in range(1, self._epochs):
      for iter, data in enumerate(loader):  # Iterate in batches over the training dataset.
        try:
          out = self._model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        except IndexError:
          print(iter)
          print(data)
          raise
        loss = self._criterion(out[:, 0], data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        self._optimizer.step()  # Update parameters based on gradients.
        self._optimizer.zero_grad()  # Clear gradients.
      train_mse.append(float(self.test(loader)[0]))
      train_losses.append(float(loss))
    return train_mse, train_losses

  def test(self, loader):
    self._model.eval()
    predictions = []
    labels = []
    for iter, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
      batch_size = data.batch.size().numel()
      out = self._model(data.x, data.edge_index, data.batch)
      predictions.append(out[:, 0].cpu().detach().numpy())
      labels.append(data.y.cpu().detach().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    mse = MseWrapper(predictions, labels)
    mse_scaled = MseWrapper(predictions, labels, scale=True)
    return mse, mse_scaled

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    sample_id = element['sample_id']
    val_mse = 0.0
    val_mse_scaled = 0.0
    test_mse = 0.0
    test_mse_scaled = 0.0
    try:
      mses, losses = self.train(element['torch_dataset']['train'])
      total_mse_val, total_mse_scaled_val = self.test(element['torch_dataset']['tuning'])
      total_mse_test, total_mse_scaled_test = self.test(element['torch_dataset']['test'])
      val_mse = float(total_mse_val)
      val_mse_scaled = float(total_mse_scaled_val)
      test_mse = float(total_mse_test)
      test_mse_scaled = float(total_mse_scaled_test)
    except Exception:
      logging.info(f'Failed to run for sample id {sample_id}')
      losses = None

    return {'losses': losses,
            'val_metrics': {'mse': val_mse,
                            'mse_scaled': val_mse_scaled},
            'test_metrics': {'mse': test_mse,
                             'mse_scaled': test_mse_scaled}}


class LRGraphBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class=None,
               benchmark_params=None, h_params=None):
    super().__init__(generator_config, model_class, benchmark_params,
                     h_params)
    self._model_name = 'LR'

  def Benchmark(self, element,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    reg = LinearRegression().fit(
      element['numpy_dataset']['train']['X'],
      element['numpy_dataset']['train']['y'])
    y_pred_val = reg.predict(element['numpy_dataset']['tuning']['X'])
    y_pred_test = reg.predict(element['numpy_dataset']['test']['X'])
    y_val = element['numpy_dataset']['tuning']['y']
    y_test = element['numpy_dataset']['test']['y']
    val_mse = MseWrapper(y_pred_val, y_val)
    val_mse_scaled = MseWrapper(y_pred_val, y_val, scale=True)
    test_mse = MseWrapper(y_pred_test, y_test)
    test_mse_scaled = MseWrapper(y_pred_test, y_test, scale=True)
    return {'losses': [],
            'val_metrics': {'mse': val_mse,
                            'mse_scaled': val_mse_scaled},
            'test_metrics': {'mse': test_mse,
                             'mse_scaled': test_mse_scaled}}

@gin.configurable
class LRGraphBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LRGraphBenchmarker(**self._h_params)

  def GetBenchmarkerClass(self):
    return LRGraphBenchmarker
