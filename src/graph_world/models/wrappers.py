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
import sklearn.metrics
import torch
from torch_geometric.nn import GAE

from .utils import MseWrapper
from .models import GCNNodeModel, GCNGraphModel, PyGBasicGraphModel
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
class NNGraphBenchmark(BenchmarkerWrapper):

  def __init__(self, model_type, benchmark_params, h_params):
    self._model_class = model_type
    self._benchmark_params = benchmark_params
    self._model_hparams = h_params

  def GetBenchmarker(self):
    return NNGraphBenchmarker(self._model_type, self._benchmark_params, self._model_hparams)

  def GetBenchmarkerClass(self):
    return NNGraphBenchmarker

  def GetModelClass(self):
    return self._model_class

  def GetModelHparams(self):
    return self._model_hparams

  def GetBenchmarkParams(self):
    return self._benchmark_params


class NNGraphBenchmarker(Benchmarker):

  def __init__(self, model_type, benchmark_params, h_params):
    self._epochs = benchmark_params['epochs']
    self._lr = benchmark_params['lr']

    self._model = PyGBasicGraphModel(model_type, h_params)
    self._optimizer = torch.optim.Adam(self._model.parameters(), self._lr, weight_decay=5e-4)
    self._criterion = torch.nn.MSELoss()
    self._model_name = model_type.__name__

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
    sample_id = element['sample_id']
    mses, losses = self.train(element['torch_dataset']['train'])
    test_mse = 0.0
    test_mse_scaled = 0.0
    try:
      total_mse, total_mse_scaled = self.test(element['torch_dataset']['test'])
      test_mse = float(total_mse)
      test_mse_scaled = float(total_mse_scaled)
    except Exception as e:
      logging.info(f'Failed to compute test mse for sample id {sample_id}')
      raise Exception(f'Failed to compute test accuracy for sample id {sample_id}') from e

    return {'losses': losses,
            'test_metrics': {'test_mse': test_mse,
                             'test_mse_scaled': test_mse_scaled}}


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


# general benchmarkers
class NNNodeBenchmarker(Benchmarker):
  def __init__(self, model_type, benchmark_params, h_params):
    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']

    self._model = model_type(**h_params)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01, weight_decay=5e-4)
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None
    self._model_name = model_type.__name__

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
    pred = out[self._test_mask].detach().numpy()

    pred_best = pred.argmax(-1)
    correct = data.y[self._test_mask].numpy()
    n_classes = out.shape[-1]
    pred_onehot = np.zeros((len(pred_best), n_classes))
    pred_onehot[np.arange(pred_best.shape[0]), pred_best] = 1

    correct_onehot = np.zeros((len(correct), n_classes))
    correct_onehot[np.arange(correct.shape[0]), correct] = 1

    results = {
        'test_accuracy': sklearn.metrics.accuracy_score(correct, pred_best),
        'test_f1_micro': sklearn.metrics.f1_score(correct, pred_best,
                                                  average='micro'),
        'test_f1_macro': sklearn.metrics.f1_score(correct, pred_best,
                                                  average='macro'),
        'test_rocauc_ovr': sklearn.metrics.roc_auc_score(correct_onehot,
                                                         pred_onehot,
                                                         multi_class='ovr'),
        'test_rocauc_ovo': sklearn.metrics.roc_auc_score(correct_onehot,
                                                         pred_onehot,
                                                         multi_class='ovo'),
        'logloss': sklearn.metrics.log_loss(correct, pred)}
    return results

  def train(self, data):
    losses = []
    for epoch in range(self._epochs):
      losses.append(float(self.train_step(data)))
    return losses

  def Benchmark(self, element):
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
      'skipped': skipped,
      'results': None
    }
    out.update(element)
    out['losses'] = None
    out['test_metrics'] = {}

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    train_mask, val_mask, test_mask = masks

    self.SetMasks(train_mask, val_mask)

    losses = self.train(torch_data)
    test_accuracy = {
        'test_accuracy': 0,
        'test_f1_micro': 0,
        'test_f1_macro': 0,
        'test_rocauc_ovr': 0,
        'test_rocauc_ovo': 0,
        'logloss': 0
    }
    try:
      # Divide by zero sometimes happens with the ksample masks.
      test_accuracy = self.test(torch_data)
    except Exception as e:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      raise Exception(
          f'Failed to compute test accuracy for sample id {sample_id}') from e

    out['losses'] = losses
    out['test_metrics'].update(test_accuracy)
    return out

@gin.configurable
class NNNodeBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return NNNodeBenchmarker(self._model_type, self._benchmark_params, self._model_hparams)

  def GetBenchmarkerClass(self):
    return NNNodeBenchmarker


# Link prediction
class LPBenchmarker(Benchmarker):
  def __init__(self, model_type, benchmark_params, h_params):
    # remove meta entries from h_params
    self._epochs = benchmark_params['epochs']

    self._model = model_type(**h_params)
    self._lp_wrapper_model = GAE(self._model)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01,
                                       weight_decay=5e-4)
    self._model_name = model_type.__name__

  def train_step(self, data):
    self._lp_wrapper_model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    z = self._model(data.x, data.train_pos_edge_index)
    loss = self._lp_wrapper_model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()  # Derive gradients.
    self._optimizer.step()  # Update parameters based on gradients.
    return loss

  def test(self, data):
    self._lp_wrapper_model.eval()
    results = {}
    z = self._model(data.x, data.train_pos_edge_index)
    roc_auc_score, average_precision_score = self._lp_wrapper_model.test(z,
                                                                         data.test_pos_edge_index,
                                                                         data.test_neg_edge_index)
    results['test_rocauc'] = roc_auc_score
    results['test_ap'] = average_precision_score
    return results

  def train(self, data):
    losses = []
    for epoch in range(self._epochs):
      losses.append(float(self.train_step(data)))
    return losses

  def Benchmark(self, element):
    torch_data = element['torch_data']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {
        'skipped': skipped,
        'results': None
    }
    out.update(element)
    out['losses'] = None
    out['test_metrics'] = {}

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    losses = self.train(torch_data)
    test_accuracy = {
        'test_rocaus': 0,
        'test_ap': 0,
    }
    try:
      # Divide by zero sometimes happens with the ksample masks.
      test_accuracy = self.test(torch_data)
    except Exception as e:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      raise Exception(
          f'Failed to compute test accuracy for sample id {sample_id}') from e

    out['losses'] = losses
    out['test_metrics'].update(test_accuracy)
    return out


@gin.configurable
class LPBenchmark(BenchmarkerWrapper):

  def GetBenchmarker(self):
    return LPBenchmarker(self._model_type, self._benchmark_params,
                       self._model_hparams)

  def GetBenchmarkerClass(self):
    return LPBenchmarker
