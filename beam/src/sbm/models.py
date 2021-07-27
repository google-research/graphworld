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

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinearGCNModel(torch.nn.Module):
  def __init__(self, num_features, num_classes, hidden_channels):
    super(LinearGCNModel, self).__init__()
    torch.manual_seed(12345)
    self.conv1 = GCNConv(num_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, num_classes)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    return x


class LinearGCN():
  def __init__(self, num_features, num_classes, hidden_channels, train_mask, val_mask, test_mask):
    self._model = LinearGCNModel(num_features, num_classes, hidden_channels)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=5e-4)
    self._criterion = torch.nn.CrossEntropyLoss()
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

  def train(self, epochs, data):
    losses = {}
    for epoch in range(epochs):
      loss = self.train_step(data)
      losses['{0:05}'.format(epoch + 1)] = '{0}'.format(loss)

    return losses
