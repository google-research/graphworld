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

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class PyGBasicGraphModel(torch.nn.Module):
  """This model wraps all basic_gnn models with a global pooling layer,
  turning them into graph models."""
  def __init__(self, model_class, h_params):
    super(PyGBasicGraphModel, self).__init__()

    # Make sure no final output conversion has been requested
    assert 'out_channels' not in h_params
    # h_params['out_channels'] = None

    # Instantiate and pass hparams to inner model
    self.inner_model_ = model_class(**h_params)

    # output
    self.lin = Linear(h_params['hidden_channels'], 1)

  def forward(self, x, edge_index, batch):
    # defer to inner forward
    x = self.inner_model_(x, edge_index)

    # apply global pooling over nodes
    x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

    # convert down to 1 dimension after pooling over nodes
    x = self.lin(x)
    return x