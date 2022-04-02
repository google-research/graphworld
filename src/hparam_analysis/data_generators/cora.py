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