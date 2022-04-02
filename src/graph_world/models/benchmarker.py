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

import json
import math

from abc import ABC, abstractmethod
import apache_beam as beam
import gin
import numpy as np

from .utils import ComputeNumPossibleConfigs, SampleModelConfig, GetCartesianProduct


class Benchmarker(ABC):

  def __init__(self, generator_config,
               model_class=None, benchmark_params=None, h_params=None):
    self._model_name = model_class.__name__ if model_class is not None else ''
    self._model_class = model_class
    self._benchmark_params = benchmark_params
    self._h_params = h_params
    self.AdjustParams(generator_config)

  # Override this function if the input data affects the model architecture.
  # See NNNodeBenchmarker for an example implementation.
  def AdjustParams(self, generator_config):
    pass

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
  def Benchmark(self, element,
                test_on_val: bool = False,
                tuning_metric: str = None,
                tuning_metric_is_loss: bool = False):
    del element  # unused
    del test_on_val  # unused
    del tuning_metric  # unused
    del tuning_metric_is_loss  # unused
    return {'losses': [], 'test_metrics': {}}


class BenchmarkerWrapper(ABC):

  def __init__(self, model_class=None, benchmark_params=None, h_params=None):
    self._model_class = model_class
    self._benchmark_params = benchmark_params
    self._h_params = h_params

  @abstractmethod
  def GetBenchmarker(self):
    return Benchmarker()

  # These two functions would be unnecessary if we were using Python 3.7. See:
  #  - https://github.com/huggingface/transformers/issues/8453
  #  - https://github.com/huggingface/transformers/issues/8212
  @abstractmethod
  def GetBenchmarkerClass(self):
    return Benchmarker

  def GetModelClass(self):
    return self._model_class

  def GetModelHparams(self):
    return self._h_params

  def GetBenchmarkParams(self):
    return self._benchmark_params



class BenchmarkGNNParDo(beam.DoFn):

  # The commented lines here, and those in process, could be uncommented and
  # replace the alternate code below it, if we were using Python 3.7. See:
  #  - https://github.com/huggingface/transformers/issues/8453
  #  - https://github.com/huggingface/transformers/issues/8212
  def __init__(self, benchmarker_wrappers, num_tuning_rounds, tuning_metric,
               tuning_metric_is_loss=False, save_tuning_results=False):
    # self._benchmarkers = [benchmarker_wrapper().GetBenchmarker() for
    #                       benchmarker_wrapper in benchmarker_wrappers]
    self._benchmarker_classes = [benchmarker_wrapper().GetBenchmarkerClass() for
                                 benchmarker_wrapper in benchmarker_wrappers]
    self._model_classes = [benchmarker_wrapper().GetModelClass() for
                                 benchmarker_wrapper in benchmarker_wrappers]
    self._h_params = [benchmarker_wrapper().GetModelHparams() for
                           benchmarker_wrapper in benchmarker_wrappers]
    self._benchmark_params = [benchmarker_wrapper().GetBenchmarkParams() for
                           benchmarker_wrapper in benchmarker_wrappers]
    # /end alternate code.
    self._output_path = None
    self._num_tuning_rounds = num_tuning_rounds
    self._tuning_metric = tuning_metric
    self._tuning_metric_is_loss = tuning_metric_is_loss
    self._save_tuning_results = save_tuning_results

  def SetOutputPath(self, output_path):
    self._output_path = output_path

  def process(self, element):
    output_data = {}
    output_data.update(element['generator_config'])
    output_data['marginal_param'] = element['marginal_param']
    output_data['fixed_params'] = element['fixed_params']
    output_data.update(element['metrics'])
    output_data['skipped'] = element['skipped']
    if 'target' in element:
      output_data['target'] = element['target']
    output_data['sample_id'] = element['sample_id']

    if element['skipped']:
      yield json.dumps(output_data)

    # for benchmarker in self._benchmarkers:
    for (benchmarker_class,
         benchmark_params,
         model_class,
         h_params) in zip(self._benchmarker_classes,
                          self._benchmark_params,
                          self._model_classes,
                          self._h_params):
      print(f'Running {benchmarker_class} and model f{model_class}')
      num_possible_configs = ComputeNumPossibleConfigs(benchmark_params, h_params)
      num_tuning_rounds = min(num_possible_configs, self._num_tuning_rounds)

      if num_tuning_rounds == 1 or self._tuning_metric == '':
        benchmark_params_sample, h_params_sample = SampleModelConfig(benchmark_params,
                                                                     h_params)
        benchmarker = benchmarker_class(element['generator_config'],
                                        model_class,
                                        benchmark_params_sample,
                                        h_params_sample)
        benchmarker_out = benchmarker.Benchmark(element,
                                                tuning_metric=self._tuning_metric,
                                                tuning_metric_is_loss=self._tuning_metric_is_loss)
        val_metrics = benchmarker_out['val_metrics']
        test_metrics = benchmarker_out['test_metrics']

      else:
        configs = []
        val_metrics_list = []
        test_metrics_list = []
        full_product = False
        if num_tuning_rounds == 0:
          num_tuning_rounds = 1
          if benchmark_params is None:
            benchmark_params_product = []
          else:
            benchmark_params_product = list(GetCartesianProduct(benchmark_params))
          num_benchmark_configs = len(benchmark_params_product)
          if num_benchmark_configs > 0:
            num_tuning_rounds *= num_benchmark_configs
          if h_params is None:
            h_params_product = []
          else:
            h_params_product = list(GetCartesianProduct(h_params))
          num_h_configs = len(h_params_product)
          if num_h_configs > 0:
            num_tuning_rounds *= num_h_configs
          full_product = True
        for i in range(num_tuning_rounds):
          if full_product:
            if num_benchmark_configs > 0:
              benchmark_index = math.floor(i / num_h_configs)
              benchmark_params_sample = benchmark_params_product[benchmark_index]
            else:
              benchmark_params_sample = None
            if num_h_configs > 0:
              h_index = i % num_h_configs
              h_params_sample = h_params_product[h_index]
            else:
              h_params_sample = None
          else:
            benchmark_params_sample, h_params_sample = SampleModelConfig(benchmark_params,
                                                                         h_params)
          benchmarker = benchmarker_class(element['generator_config'],
                                          model_class,
                                          benchmark_params_sample,
                                          h_params_sample)
          benchmarker_out = benchmarker.Benchmark(element,
                                                  tuning_metric=self._tuning_metric,
                                                  tuning_metric_is_loss=self._tuning_metric_is_loss)
          configs.append((benchmark_params_sample, h_params_sample))
          val_metrics_list.append(benchmarker_out['val_metrics'])
          test_metrics_list.append(benchmarker_out['test_metrics'])

        val_scores = [metrics[self._tuning_metric] for metrics in val_metrics_list]
        test_scores = [metrics[self._tuning_metric] for metrics in test_metrics_list]
        if self._tuning_metric_is_loss:
          best_tuning_round = np.argmin(val_scores)
        else:
          best_tuning_round = np.argmax(val_scores)
        benchmark_params_sample, h_params_sample = configs[best_tuning_round]
        output_data['%s__num_tuning_rounds' % benchmarker.GetModelName()] = num_tuning_rounds
        if self._save_tuning_results:
          output_data['%s__configs' % benchmarker.GetModelName()] = configs
          output_data['%s__val_scores' % benchmarker.GetModelName()] = val_scores
          output_data['%s__test_scores' % benchmarker.GetModelName()] = test_scores

        val_metrics = val_metrics_list[best_tuning_round]
        test_metrics = test_metrics_list[best_tuning_round]

      # Return benchmark data for next beam stage.

      for key, value in val_metrics.items():
        output_data[f'{benchmarker.GetModelName()}__val_{key}'] = value
      for key, value in test_metrics.items():
        output_data[f'{benchmarker.GetModelName()}__test_{key}'] = value


      if benchmark_params_sample is not None:
        for key, value in benchmark_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}__train_{key}'] = value

      if h_params_sample is not None:
        for key, value in h_params_sample.items():
          output_data[f'{benchmarker.GetModelName()}__model_{key}'] = value

    yield json.dumps(output_data)