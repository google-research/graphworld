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

# Method to get a single val/test result from GCN
from graph_world.models.basic_gnn import GCN
import numpy as np
import torch


def test_gcn(data,
             hidden_channels=16,
             weight_decay=5e-4,
             lr=0.01,
             dropout=0.5,
             num_layers=1,
             decrease_tol=50,
             verbose=False):
  model = GCN(in_channels=np.squeeze(data.x.numpy()).shape[1],
              hidden_channels=hidden_channels,
              num_layers=num_layers,
              out_channels=len(set(data.y.numpy())),
              dropout=dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
  criterion = torch.nn.CrossEntropyLoss()

  def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    out = out[data.label_mask]
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    out = out[data.label_mask]
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return test_acc

  decrease_tol = decrease_tol
  best_val_acc = 0.0
  best_test_acc = 0.0
  num_decreases = 0

  epoch_count = 0

  while True:
    loss = train()
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)

    if val_acc <= best_val_acc:
      num_decreases += 1
    else:
      num_decreases = 0

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_test_acc = test_acc

    if verbose:
      print(
        f'Epoch: {epoch_count:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Num Decreases: {num_decreases:d}')

    epoch_count += 1

    if num_decreases == decrease_tol:
      break

  return best_val_acc, best_test_acc, epoch_count