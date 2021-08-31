import gin

from abc import ABC, abstractmethod

import apache_beam as beam

class Benchmarker(ABC):
  # Train and test the model.
  # Arguments:
  #   * element: output of the 'Convert to torchgeo' beam stage.
  #   * output_path: where to save logs and data.
  # Returns:
  #   * named dict with keys/vals:
  #      'losses': iterable of loss values over the epochs.
  #      'test_metrics': dict of named test metrics for the benchmark run.
  @abstractmethod
  def Benchmark(self, element):
    del element # unused
    return {'losses': [], 'test_metrics': {}}