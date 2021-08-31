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

"""Beam pipeline for generating random graph samples.
"""
import argparse
import json
import logging
import os
import setuptools

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import numpy as np
import pandas as pd
import gin

# Generator-agnostic imports
from generator_beam_handler import GeneratorBeamHandlerWrapper
from generator_config_sampler import ParamSamplerSpec
from models.wrappers import LinearGCNWrapper

# Generator-specific imports
from sbm.beam_handler import SbmBeamHandler
from substructure.beam_handler import SubstructureBeamHandler
from substructure.simulator import Substructure


def CombineDataframes(dfs):
  return pd.concat(dfs or [pd.DataFrame()])

def WriteDataFrame(df, output_path):
  df.to_csv(os.path.join(output_path, "results_df.csv"))

def main(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument('--output',
                      dest='output',
                      default='/tmp/graph_configs.json',
                      help='Location to write output files.')

  parser.add_argument('--gin_config',
                      dest='gin_config',
                      default='',
                      help='Location of gin config (/app/configs = /src/configs).')

  args, pipeline_args = parser.parse_known_args(argv)
  logging.info(f'Pipeline Args: {pipeline_args}')
  gin.parse_config_file(args.gin_config)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  gen_handler_wrapper = GeneratorBeamHandlerWrapper()
  gen_handler_wrapper.SetOutputPath(args.output)

  with beam.Pipeline(options=pipeline_options) as p:
    graph_samples = (
        p
        | 'Create Sample Ids' >> beam.Create(range(gen_handler_wrapper.nsamples))
        | 'Sample Graphs' >> beam.ParDo(gen_handler_wrapper.handler.GetSampleDoFn())
    )

    graph_samples | 'Write Sampled Graph' >> beam.ParDo(
      gen_handler_wrapper.handler.GetWriteDoFn())

    torch_data = (
        graph_samples | 'Compute graph metrics.' >> beam.ParDo(
      gen_handler_wrapper.handler.GetGraphMetricsParDo())
                      | 'Convert to torchgeo data.' >> beam.ParDo(
      gen_handler_wrapper.handler.GetConvertParDo())
    )

    (torch_data | 'Filter skipped conversions' >> beam.Filter(lambda el: el['skipped'])
     | 'Extract skipped sample ids' >> beam.Map(lambda el: el['sample_id'])
     | 'Write skipped text file' >> beam.io.WriteToText(os.path.join(args.output, 'skipped.txt')))

    dataframe_rows = (
        torch_data | 'Benchmark Simple GCN.' >> beam.ParDo(
      gen_handler_wrapper.handler.GetBenchmarkParDo()))

    (dataframe_rows | 'Combine into single dataframe.' >> beam.CombineGlobally(CombineDataframes)
                    | 'Write dataframe.' >> beam.Map(WriteDataFrame, args.output))


if __name__ == '__main__':
  main(None)
