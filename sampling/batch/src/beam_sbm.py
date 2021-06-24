"""Beam pipeline for generating random graph samples.
"""
import argparse
import os
import json
import setuptools

# from absl import app
from absl import logging
# from absl import flags

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import numpy as np


class SampleSbmDoFn(beam.DoFn):

    def process(self, element):
        """Sample and save SMB outputs given a configuration filepath.
        """
        # Avoid save_main_session in Pipeline args so the controller doesn't
        # have to import the same libraries as the workers which may be using
        # a custom container. The import will execute once then the sys.modeules
        # will be referenced to further calls.
        import numpy as np
        from sbm.sbm_simulator import StochasticBlockModel

        # Parameterize me...
        nvertex_min = 5
        nvertex_max = 50
        nedges_min = 10
        nedges_max = 100
        feature_dim_min = 16
        feature_dim_max = 16
        edge_center_distance_min = 1.0
        edge_center_distance_max = 10.0
        edge_feature_dim_min = 4
        edge_feature_dim_max = 4

        num_vertices = np.random.randint(nvertex_min, nvertex_max+1)
        num_edges = np.random.randint(nedges_min, nedges_max+1)
        feature_dim = np.random.randint(feature_dim_min, feature_dim_max+1)
        edge_center_distance = np.random.uniform(edge_center_distance_min, edge_center_distance_max)
        edge_feature_dim = np.random.randint(edge_feature_dim_min, edge_feature_dim_max+1)

        generator_config = {
            'generator_name': 'StochasticBlockModel',
            'num_verticies': num_vertices,
            'num_edges': num_edges,
            'feature_dim': feature_dim,
            'edge_center_distance': edge_center_distance,
            'edge_feature_dim': edge_feature_dim
        }

        data = GenerateStochasticBlockModelWithFeatures(
            num_verticies = generator_config['num_verticies'],
            num_edges = generator_config['num_edges'],
            pi = np.array([0.25, 0.25, 0.25, 0.25]),
            prop_mat = np.ones((4, 4)) + 9.0 * np.diag([1,1,1,1]),
            feature_center_distance = generator_config['feature_center_distance'],
            edge_center_distance = generator_config['edge_center_distance'],
            ege_feature_dim = generator_config['edge_feature_dim']
        )

        yield generator_config

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--output',
                        dest='output',
                        default='/tmp/graph_configs.json',
                        help='Location to write output files.')

    args, pipeline_args = parser.parse_known_args(argv)

    logging.info(f'output: {args.output}')
    logging.info(f'Pipeline Aargs: {pipeline_args}')
    print(f'Pipeline Args: {pipeline_args}')

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = False

    with beam.Pipeline(options=pipeline_options) as p:

        results = (
            p
            | 'Create Inputs' >> beam.Create(range(100))
            | 'Sample Graphs' >> beam.ParDo(SampleSbmDoFn())
            | 'Write Configs' >> beam.io.WriteToText(args.output)
        )

if __name__ == '__main__':
    # flags.mark_flag_as_required('input_collection')
    # flags.mark_flag_as_required('output_path')
    # app.run(main)
    main(None)
