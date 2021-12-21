from torch_geometric.datasets import Planetoid
import numpy as np
import torch

def get_cora():
  dataset = Planetoid(root='/tmp/Cora', name='Cora')
  data = dataset[0]

  # Row-normalize features
  X = np.squeeze(data.x.numpy())
  data.x = torch.tensor((X.T / np.sum(X, axis=1)).T)
  return data