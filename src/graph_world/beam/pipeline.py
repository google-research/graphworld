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

import argparse
import logging
import os
import sys

import apache_beam as beam
import pandas as pd
import gin

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

# Generator-agnostic imports
from ..beam.generator_beam_handler import GeneratorBeamHandlerWrapper
from ..beam.generator_config_sampler import ParamSamplerSpec
from .task_benchmarkers import *

# Generator-specific imports
from ..nodeclassification.beam_handler import NodeClassificationBeamHandler
from ..graphregression.beam_handler import GraphRegressionBeamHandler
from ..linkprediction.beam_handler import LinkPredictionBeamHandler
from ..noderegression.beam_handler import NodeRegressionBeamHandler


def entry(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument('--output',
                      dest='output',
                      default='/tmp/graph_configs.json',
                      help='Location to write output files.')

  parser.add_argument('--gin_files',
                      dest='gin_files',
                      nargs='+',
                      type=str,
                      help='Paths to config files.')
  
  parser.add_argument('--gin_params',
                      dest='gin_params',
                      nargs='+',
                      type=str,
                      help='Gin config bindings.')

  parser.add_argument('--write_intermediate',
                      dest='write_samples',
                      default=False,
                      help='Whether to write sampled graph data. Saves CPU and disk if disabled.')

  args, pipeline_args = parser.parse_known_args(argv)
  sys.stdout.flush()
  print(f'Pipeline Args: {pipeline_args}', flush=True)
  print(f'Binary Args: {args}', flush=True)
  gin.parse_config_files_and_bindings(args.gin_files, args.gin_params)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  gen_handler_wrapper = GeneratorBeamHandlerWrapper()
  gen_handler_wrapper.SetOutputPath(args.output)

  with beam.Pipeline(options=pipeline_options) as p:
    graph_samples = (
        p
        | 'Create Sample Ids' >> beam.Create(
        range(gen_handler_wrapper.nsamples))
        | 'Sample Graphs' >> beam.ParDo(
        gen_handler_wrapper.handler.GetSampleDoFn())
    )
    if args.write_samples:
      graph_samples | 'Write Sampled Graph' >> beam.ParDo(
          gen_handler_wrapper.handler.GetWriteDoFn())

    torch_data = (
        graph_samples | 'Compute graph metrics.' >> beam.ParDo(
        gen_handler_wrapper.handler.GetGraphMetricsParDo())
        | 'Convert to torchgeo data.' >> beam.ParDo(
        gen_handler_wrapper.handler.GetConvertParDo())
    )

    (torch_data | 'Filter skipped conversions' >> beam.Filter(
        lambda el: el['skipped'])
     | 'Extract skipped sample ids' >> beam.Map(lambda el: el['sample_id'])
     | 'Write skipped text file' >> beam.io.WriteToText(
            os.path.join(args.output, 'skipped.txt')))

    dataframe_rows = (
        torch_data | 'Benchmark Simple GCN.' >> beam.ParDo(
        gen_handler_wrapper.handler.GetBenchmarkParDo()))

    dataframe_rows | 'Write JSON' >> beam.io.WriteToText(
        os.path.join(args.output, 'results.ndjson'), num_shards=10)
