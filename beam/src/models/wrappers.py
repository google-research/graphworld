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

import numpy as np
import torch

from models.models import LinearGCNModel, LinearGraphGCNModel


class LinearGCN:
  def __init__(self, num_features, num_classes, hidden_channels, epochs):
    self._model = LinearGCNModel(num_features, num_classes, hidden_channels)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=5e-4)
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None
    self._epochs = epochs

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


class LinearGraphGCN:

  def __init__(self, num_features, hidden_channels=64, epochs=100, lr=0.0001):
    self._model = LinearGraphGCNModel(num_features, hidden_channels)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
    self._criterion = torch.nn.MSELoss()
    self._epochs = epochs

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
          loss = self._criterion(out, data.y)  # Compute the loss.
          loss.backward()  # Derive gradients.
          self._optimizer.step()  # Update parameters based on gradients.
          self._optimizer.zero_grad()  # Clear gradients.
      train_mse.append(float(self.test(loader)))
      train_losses.append(float(loss))
    return train_mse, train_losses

  def test(self, loader):
      self._model.eval()
      total_mse = 0.0
      label_variance = 0.0
      for iter, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
        batch_size = data.batch.size().numel()
        out = self._model(data.x, data.edge_index, data.batch)
        total_mse += float(self._criterion(out, data.y)) * batch_size
        label_variance += np.std(data.y.numpy()) ** 2.0 * batch_size
      return total_mse / label_variance