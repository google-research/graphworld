import logging
import json
import os

import apache_beam as beam

from models.wrappers import LinearGCN

class BenchmarkGNNParDo(beam.DoFn):
  def __init__(self, output_path, num_features, num_classes, hidden_channels, epochs):
    self._output_path = output_path
    self._num_features = num_features
    self._num_classes = num_classes
    self._hidden_channels = hidden_channels
    self._epochs = epochs

  def process(self, element):
    sample_id = element['sample_id']
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
    linear_model = LinearGCN(
      self._num_features,
      self._num_classes,
      self._hidden_channels,
      train_mask,
      val_mask,
      test_mask)

    losses = linear_model.train(self._epochs, torch_data)
    test_accuracy = None
    try:
      # Divide by zero somesimtes happens with the ksample masks.
      test_accuracy = linear_model.test(torch_data)
    except:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')

    results = {
      'sample_id': sample_id,
      'losses': losses,
      'test_accuracy': test_accuracy,
      'generator_config': element['generator_config']
    }

    results_object_name = os.path.join(self._output_path, '{0:05}_results.txt'.format(sample_id))
    with beam.io.filesystems.FileSystems.create(results_object_name, 'text/plain') as f:
      buf = bytes(json.dumps(results), 'utf-8')
      f.write(buf)
      f.close()

    yield results