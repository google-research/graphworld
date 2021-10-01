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
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class GCNNodeModel(torch.nn.Module):
  def __init__(self, num_features, num_classes, hidden_channels):
    super(GCNNodeModel, self).__init__()
    torch.manual_seed(12345)
    self.conv1 = GCNConv(num_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, num_classes)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    return x


class GCNGraphModel(torch.nn.Module):
  def __init__(self, num_features, hidden_channels):
    super(GCNGraphModel, self).__init__()
    torch.manual_seed(12345)
    self.conv1 = GCNConv(num_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.lin = Linear(hidden_channels, 1)

  def forward(self, x, edge_index, batch):
    # 1. Obtain node embeddings
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = self.conv2(x, edge_index)

    # 2. Readout layer
    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

    # 3. Apply a final classifier
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)

    return x

class PyGBasicModel(torch.nn.Module):
  def __init__(self, model_type, *h_params):
    # Instantiate and pass hparams to inner model
    self.inner_model_ = model_type(h_params)

  def forward(self, x, edge_index, batch):
    # defer to inner forward
    return self.inner_model_.forward(x, edge_index, batch)