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

import apache_beam as beam
import pandas as pd
import gin

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from ..utils.config_enumeration import enumerate_configs
from ..beam.hparam_eval import HparamBeamHandler


def entry(argv=None):
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_path',
                      dest='dataset_path',
                      default='',
                      help=('Location of input data files. '
                            'Default behavior downloads data from web. '
                            'GCP runs will need to input GCS dataset path. '))

  parser.add_argument('--dataset_name',
                      dest='dataset_name',
                      default='',
                      help=('Name of the dataset. '
                            'dataset_name.npz must be in dataset_path. '))

  parser.add_argument('--output',
                      dest='output',
                      default='/tmp/graph_configs.json',
                      help='Location to write output files.')

  parser.add_argument('--gcp_pname',
                      dest='gcp_pname',
                      default='research-graph')

  parser.add_argument('--gcs_auth',
                      dest='gcs_auth',
                      default=None)

  parser.add_argument('--sim', dest='sim', action='store_true')
  parser.add_argument('--no-sim', dest='sim', action='store_false')
  parser.set_defaults(sim=True)

  parser.add_argument('--gin_config',
                      dest='gin_config',
                      default='',
                      help='Location of gin config (/app/configs = /src/configs).')

  args, pipeline_args = parser.parse_known_args(argv)
  logging.info(f'Pipeline Args: {pipeline_args}')
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  gin.parse_config_file(args.gin_config)
  hparam_handler = HparamBeamHandler(dataset_path=args.dataset_path,
                                     sim=args.sim,
                                     dataset_name=args.dataset_name,
                                     project_name=args.gcp_pname,
                                     gcs_auth=args.gcs_auth)
  logging.info("----about to run pipeline")
  with beam.Pipeline(options=pipeline_options) as p:
    dataframe_rows = (
      p | 'Enumerate hyperparameter gridpoints.' >> beam.Create(
        enumerate_configs())
        | 'Test GCN.' >> beam.ParDo(hparam_handler.GetGcnTester())
        | 'Write JSON' >> beam.io.WriteToText(
        os.path.join(args.output, 'results.ndjson'), num_shards=10))

