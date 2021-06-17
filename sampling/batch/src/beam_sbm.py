"""Beam pipeline for generating random graph samples.
"""

import os

from absl import app
from absl import logging
from absl import flags

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

FLAGS = flags.FLAGS

# Running beam workers that are setup via docker container:
#    https://cloud.google.com/dataflow/docs/guides/using-custom-containers

# Pipeline config.
flags.DEFINE_string('input_collection', None, 'Path to text file containing the input collection.')
flags.DEFINE_string('output_path', None, 'Path for output artifacts.')

# Beam config.
flags.DEFINE_string('runner', 'DirectRunner', 'Change to DataflowRunner to run on GCP.')
flags.DEFINE_string('environment_config', None, 'Environment config, e.g., docker container URI.')
flags.DEFINE_string('environment_type', 'Docker', 'For setting up flume workers via docker container.')
flags.DEFINE_string('job_name', None, 'SbmBeamSampler')

# GCP config.
flags.DEFINE_string('project_id', None, 'GCP project ID.')
flags.DEFINE_string('region', None, 'GCP region ID.')
flags.DEFINE_string('staging_location', None, 'GCP staging location, e.g.,  gs://YOUR_BUCKET/STAGING_DIR')
flags.DEFINE_string('temp_location', None, 'GCP temp location, e.g., gs://YOUR_BUCKET/TEMP_LOCATION')



class SampleSmbDoFn(beam.DoFn):
    def process(self, config_file, output_path):
        """Sample and save SMB outputs given a configuration filepath.
        """
        # TODO (switch on gcp vs. local file)

def main(argv):

    pipeline_args = {
        f"--runner={FLAGS.runner}",
    }

    logging.info(pipeline_args)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:

        config_files = p | ReadFromText(FLAGS.input_collection)

if __name__ == '__main__':
    flags.mark_flag_as_required('input_collection')
    flags.mark_flag_as_required('output_path')
    app.run(main)
