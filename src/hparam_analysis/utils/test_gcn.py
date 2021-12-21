# Method to get a single val/test result from GCN
import numpy as np

import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_channels,
                 dropout=0.5):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def test_gcn(data,
             hidden_channels=16,
             weight_decay=5e-4,
             lr=0.01,
             dropout=0.5,
             decrease_tol=50,
             verbose=False):
  model = GCN(num_features=np.squeeze(data.x.numpy()).shape[1],
              num_classes=len(set(data.y.numpy())),
              hidden_channels=hidden_channels,
              dropout=dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
  criterion = torch.nn.CrossEntropyLoss()

  def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
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