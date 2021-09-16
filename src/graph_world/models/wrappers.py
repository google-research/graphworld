# Copyright 2021 Google LLC
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
from sklearn.metrics import mean_squared_error
import torch

from .utils import MseWrapper
from .models import GCNNodeModel, GCNGraphModel
from .benchmarker import Benchmarker, BenchmarkerWrapper

class GCNNodeBenchmarker(Benchmarker):
  def __init__(self, num_features, num_classes, hidden_channels, epochs, model_name):
    self._model = GCNNodeModel(num_features, num_classes, hidden_channels)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=5e-4)
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None
    self._epochs = epochs
    self._model_name = model_name

  def SetMasks(self, train_mask, val_mask):
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = val_mask

  def train_step(self, data):
    self._model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    out = self._model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = self._criterion(out[self._train_mask],
                           data.y[self._train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data):
    self._model.eval()
    out = self._model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[self._test_mask] == data.y[self._test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(self._test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

  def train(self, data):
    losses = []
    for epoch in range(self._epochs):
      losses.append(float(self.train_step(data)))
    return losses

  def Benchmark(self, element):
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']

    out = {
      'skipped': skipped,
      'results': None
    }

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return

    train_mask, val_mask, test_mask = masks

    self.SetMasks(train_mask, val_mask)

    losses = self.train(torch_data)
    test_accuracy = 0.0
    try:
      # Divide by zero somesimtes happens with the ksample masks.
      test_accuracy = self.test(torch_data)
    except:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')

    return {'losses': losses,
            'test_metrics': {'test_accuracy': test_accuracy}}


@gin.configurable
class NodeGCN(BenchmarkerWrapper):

  def __init__(self, num_features, num_classes, hidden_channels, epochs, model_name):
    self._model_hparams = {
      "num_features": num_features,
      "num_classes": num_classes,
      "hidden_channels": hidden_channels,
      "epochs": epochs,
      "model_name": model_name
    }

  def GetBenchmarker(self):
    return GCNNodeBenchmarker(**self._model_hparams)

  def GetBenchmarkerClass(self):
    return GCNNodeBenchmarker

  def GetModelHparams(self):
    return self._model_hparams


class GCNGraphBenchmarker(Benchmarker):

  def __init__(self, num_features, hidden_channels=64, epochs=100, lr=0.0001, model_name=''):
    self._model = GCNGraphModel(num_features, hidden_channels)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
    self._criterion = torch.nn.MSELoss()
    self._epochs = epochs
    self._model_name = model_name

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

  def Benchmark(self, element):
    mses, losses = self.train(element['torch_dataset']['train'])
    test_mse = 0.0
    test_mse_scaled = 0.0
    try:
      total_mse, total_mse_scaled = self.test(element['torch_dataset']['test'])
      test_mse = float(total_mse)
      test_mse_scaled = float(total_mse_scaled)
    except:
      logging.info(f'Failed to compute test mse for sample id {sample_id}')

    return {'losses': losses,
            'test_metrics': {'test_mse': test_mse,
                             'test_mse_scaled': test_mse_scaled}}

@gin.configurable
class GraphGCN(BenchmarkerWrapper):

  def __init__(self, num_features, hidden_channels, epochs, lr, model_name):
    self._model_hparams = {
      "num_features": num_features,
      "hidden_channels": hidden_channels,
      "epochs": epochs,
      "lr": lr,
      "model_name": model_name
    }

  def GetBenchmarker(self):
    return GCNGraphBenchmarker(**self._model_hparams)

  def GetBenchmarkerClass(self):
    return GCNGraphBenchmarker

  def GetModelHparams(self):
    return self._model_hparams


class LinearGraph(Benchmarker):

  def __init__(self, model_name=''):
    self._model_name = model_name

  def Benchmark(self, element):
    reg = LinearRegression().fit(
      element['numpy_dataset']['train']['X'],
      element['numpy_dataset']['train']['y'])
    y_pred = reg.predict(
      element['numpy_dataset']['test']['X'])
    y_test = (
      element['numpy_dataset']['test']['y']
    )
    test_mse = MseWrapper(y_pred, y_test)
    test_mse_scaled = MseWrapper(y_pred, y_test, scale=True)
    return {'losses': [],
            'test_metrics': {'test_mse': test_mse,
                             'test_mse_scaled': test_mse_scaled}}

@gin.configurable
class LinearGraphWrapper(BenchmarkerWrapper):

  def __init__(self, model_name):
    self._model_hparams = {
      "model_name": model_name
    }

  def GetBenchmarker(self):
    return LinearGraph(**self._model_hparams)

  def GetBenchmarkerClass(self):
    return LinearGraph

  def GetModelHparams(self):
    return self._model_hparams