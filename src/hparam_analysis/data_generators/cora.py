from torch_geometric.datasets import Planetoid
import numpy as np
import torch

from ..data_generators.load_cora_gcp import PlanetoidGcp

def get_cora(dataset_path):
  if not dataset_path:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
  else:
    dataset = PlanetoidGcp(url=dataset_path, root='/tmp/Cora', name='Cora')
  data = dataset[0]

  # Row-normalize features
  X = np.squeeze(data.x.numpy())
  data.x = torch.tensor((X.T / np.sum(X, axis=1)).T)
  return data