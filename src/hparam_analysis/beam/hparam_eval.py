import apache_beam as beam

from ..data_generators.cora import get_cora
from ..utils.test_gcn import test_gcn

class GcnTester(beam.DoFn):

  def process(self, test_config):

    data = get_cora()
    best_val_acc, best_test_acc, epoch_count = test_gcn(
        data,
        hidden_channels=test_config['hidden_channels'],
        weight_decay=test_config['weight_decay'],
        lr=test_config['learning_rate'],
        dropout=test_config['dropout']
    )

    output = test_config
    output['best_val_acc'] = best_val_acc
    output['best_test_acc'] = best_test_acc
    output['epoch_count'] = epoch_count

    yield output
