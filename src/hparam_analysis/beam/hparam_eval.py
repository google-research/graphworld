import logging

import apache_beam as beam

import numpy as np
import torch

from ..data_generators.cora import get_cora
from ..utils.splits import get_random_split
from ..utils.test_gcn import test_gcn

class GcnTester(beam.DoFn):

  def process(self, test_config):

    output = test_config
    data = get_cora()

    splits = [get_random_split(data, seed) for
              seed in [42, 12345, 808]]

    val_accs = []
    test_accs = []
    epochs = []

    for split in splits:
      data.train_mask = torch.tensor(split[0])
      data.val_mask = torch.tensor(split[1])
      data.test_mask = torch.tensor(split[2])

      best_val_acc, best_test_acc, epoch_count = test_gcn(
          data,
          hidden_channels=test_config['hidden_channels'],
          weight_decay=test_config['weight_decay'],
          lr=test_config['learning_rate'],
          dropout=test_config['dropout']
      )

      val_accs.append(best_val_acc)
      test_accs.append(best_test_acc)
      epochs.append(epoch_count)

    output['val_acc_mean'] = np.mean(val_accs)
    output['test_acc_mean'] = np.mean(test_accs)
    output['epoch_mean'] = np.mean(epochs)
    output['val_acc_std'] = np.std(val_accs)
    output['test_acc_std'] = np.std(test_accs)
    output['epoch_std'] = np.std(epochs)

    yield output
