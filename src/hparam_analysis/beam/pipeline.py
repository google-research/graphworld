import argparse
import logging
import os

import apache_beam as beam
import pandas as pd

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from ..utils.config_enumeration import enumerate_configs


def entry(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument('--output',
                      dest='output',
                      default='/tmp/graph_configs.json',
                      help='Location to write output files.')

  args, pipeline_args = parser.parse_known_args(argv)
  logging.info(f'Pipeline Args: {pipeline_args}')
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  with beam.Pipeline(options=pipeline_options) as p:
    dataframe_rows = (
      p | 'Enumerate hyperparameter gridpoints.' >> beam.Create(
        enumerate_configs())
        | 'Write JSON' >> beam.io.WriteToText(
        os.path.join(args.output, 'results.ndjson'), num_shards=10))

