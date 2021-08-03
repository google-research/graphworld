import logging
import json
import os

import apache_beam as beam
import pandas as pd

from models.wrappers import LinearGCN

class BenchmarkGNNParDo(beam.DoFn):
  def __init__(self, num_features, num_classes, hidden_channels, epochs):
    self._num_features = num_features
    self._num_classes = num_classes
    self._hidden_channels = hidden_channels
    self._epochs = epochs

  def SetOutputPath(self, output_path):
    self._output_path = output_path

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

    benchmark_result = {
      'sample_id': sample_id,
      'losses': losses,
      'test_accuracy': test_accuracy,
      'generator_config': element['generator_config']
    }

    results_object_name = os.path.join(self._output_path, '{0:05}_results.txt'.format(sample_id))
    with beam.io.filesystems.FileSystems.create(results_object_name, 'text/plain') as f:
      buf = bytes(json.dumps(benchmark_result), 'utf-8')
      f.write(buf)
      f.close()

    test_accuracy = (0.0 if benchmark_result['test_accuracy'] is None else
                     benchmark_result['test_accuracy'])
    yield pd.DataFrame(
      data={
        "test_accuracy": test_accuracy,
        "num_vertices": benchmark_result['generator_config']['nvertex'],
        "num_edges": benchmark_result['generator_config']['nedges'],
        "feature_dim": benchmark_result['generator_config']['feature_dim'],
        "feature_center_distance": benchmark_result['generator_config']['feature_center_distance'],
        "edge_center_distance": benchmark_result['generator_config']['edge_center_distance'],
        "edge_feature_dim": benchmark_result['generator_config']['edge_feature_dim']
      },
      index=[sample_id]
    )