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
from apache_beam.dataframe.convert import to_dataframe
import numpy as np

from sbm.beam_handler import SbmBeamHandler


def main(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument('--output',
                      dest='output',
                      default='/tmp/graph_configs.json',
                      help='Location to write output files.')

  parser.add_argument('--nsamples',
                      dest='nsamples',
                      type=int,
                      default=100,
                      help='The number of graph samples.')

  parser.add_argument('--nvertex_min',
                      dest='nvertex_min',
                      type=int,
                      default=5,
                      help='Minimum number of nodes in graph samples.')

  parser.add_argument('--nvertex_max',
                      dest='nvertex_max',
                      type=int,
                      default=50,
                      help='Maximum number of nodes in the graph samples.')

  parser.add_argument('--nedges_min',
                      dest='nedges_min',
                      type=int,
                      default=10,
                      help='Minimum number of edges in the graph samples.')

  parser.add_argument('--nedges_max',
                      dest='nedges_max',
                      type=int,
                      default=100,
                      help='Maximum number of edges in the graph samples.')

  parser.add_argument('--feature_center_distance_max',
                      dest='feature_center_distance_max',
                      type=float,
                      default=10.0,
                      help=('Maximum cluster mean feature distance in the'
                            'graph samples.'))

  # Have gin specify 'generator_handler' as one of the derived GeneratorBeamHandler classes.

  args, pipeline_args = parser.parse_known_args(argv)

  logging.info(f'Pipeline Args: {pipeline_args}')

  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  # Parameterize
  num_features = 16
  num_classes = 4
  hidden_channels = 8
  epochs = 256

  generator_handler = SbmBeamHandler(
    args.output, args.nvertex_min, args.nvertex_max, args.nedges_min, args.nedges_max, args.feature_center_distance_max,
    num_features, num_classes, hidden_channels, epochs)

  def ConvertToRow(benchmark_result):
    test_accuracy = (0.0 if benchmark_result['test_accuracy'] is None else
                     benchmark_result['test_accuracy'])
    return beam.Row(
      test_accuracy=test_accuracy,
      num_vertices=benchmark_result['generator_config']['num_vertices'],
      num_edges=benchmark_result['generator_config']['num_edges'],
      feature_dim=benchmark_result['generator_config']['feature_dim'],
      feature_center_distance=benchmark_result['generator_config']['feature_center_distance'],
      edge_center_distance=benchmark_result['generator_config']['edge_center_distance'],
      edge_feature_dim=benchmark_result['generator_config']['edge_feature_dim']
    )

  with beam.Pipeline(options=pipeline_options) as p:
    graph_samples = (
        p
        | 'Create Sample Ids' >> beam.Create(range(args.nsamples))
        | 'Sample Graphs' >> beam.ParDo(generator_handler.GetSampleDoFn())
    )

    graph_samples | 'Write Sampled Graph' >> beam.ParDo(generator_handler.GetWriteDoFn())

    torch_data = (
        graph_samples | 'Convert to torchgeo data.' >> beam.ParDo(generator_handler.GetConvertParDo()))

    (torch_data | 'Filter skipped conversions' >> beam.Filter(lambda el: el['skipped'])
     | 'Extract skipped sample ids' >> beam.Map(lambda el: el['sample_id'])
     | 'Write skipped text file' >> beam.io.WriteToText(os.path.join(args.output, 'skipped.txt')))

    dataframe_rows = (
        torch_data | 'Benchmark Simple GCN.' >> beam.ParDo(generator_handler.GetBenchmarkParDo())
        | 'Convert to dataframe rows.' >> beam.Map(ConvertToRow))

    to_dataframe(dataframe_rows).to_csv(os.path.join(args.output, 'results_df.csv'))


if __name__ == '__main__':
  main(None)
