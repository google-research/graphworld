import json

from abc import ABC, abstractmethod
import apache_beam as beam
import gin


class Benchmarker(ABC):

  def __init__(self):
    self._model_name = ''

  def GetModelName(self):
    return self._model_name

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


class BenchmarkerWrapper(ABC):

  @abstractmethod
  def GetBenchmarker(self):
    return Benchmarker()

  # These two functions would be unnecessary if we were using Python 3.7. See:
  #  - https://github.com/huggingface/transformers/issues/8453
  #  - https://github.com/huggingface/transformers/issues/8212
  @abstractmethod
  def GetBenchmarkerClass(self):
    return Benchmarker

  @abstractmethod
  def GetModelHparams(self):
    return {}


class BenchmarkGNNParDo(beam.DoFn):

  # The commented lines here, and those in process, could be uncommented and
  # replace the alternate code below it, if we were using Python 3.7. See:
  #  - https://github.com/huggingface/transformers/issues/8453
  #  - https://github.com/huggingface/transformers/issues/8212
  def __init__(self, benchmarker_wrappers):
    # self._benchmarkers = [benchmarker_wrapper().GetBenchmarker() for
    #                       benchmarker_wrapper in benchmarker_wrappers]
    self._benchmarker_classes = [benchmarker_wrapper().GetBenchmarkerClass() for
                                 benchmarker_wrapper in benchmarker_wrappers]
    self._model_classes = [benchmarker_wrapper().GetModelClass() for
                                 benchmarker_wrapper in benchmarker_wrappers]
    self._model_hparams = [benchmarker_wrapper().GetModelHparams() for
                           benchmarker_wrapper in benchmarker_wrappers]
    self._benchmark_params = [benchmarker_wrapper().GetBenchmarkParams() for
                           benchmarker_wrapper in benchmarker_wrappers]
    # /end alternate code.
    self._output_path = None

  def SetOutputPath(self, output_path):
    self._output_path = output_path

  def process(self, element):
    output_data = {}
    output_data.update(element['generator_config'])
    output_data.update(element['metrics'])
    output_data['skipped'] = element['skipped']
    output_data['sample_id'] = element['sample_id']

    if element['skipped']:
      yield json.dumps(output_data)

    # for benchmarker in self._benchmarkers:
    for benchmarker_class, benchmark_params, model_class, model_hparams in zip(self._benchmarker_classes, self._benchmark_params, self._model_classes, self._model_hparams):
      print(f'Running {benchmarker_class} and model f{model_class}')
      benchmarker = benchmarker_class(model_class, benchmark_params, model_hparams)  # new benchmarker gets model and model_params
      benchmarker_out = benchmarker.Benchmark(element)

      # Return benchmark data for next beam stage.
      for key, value in benchmarker_out['test_metrics'].items():
        output_data[f'{benchmarker.GetModelName()}__{key}'] = value

    yield json.dumps(output_data)