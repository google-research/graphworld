"""Beam pipeline for generating random graph samples.
"""
import argparse
import os
import json
import setuptools

from absl import logging

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


class SampleSbmDoFn(beam.DoFn):

    def process(self, sample_id):
        """Sample and save SMB outputs given a configuration filepath.
        """
        # Avoid save_main_session in Pipeline args so the controller doesn't
        # have to import the same libraries as the workers which may be using
        # a custom container. The import will execute once then the sys.modeules
        # will be referenced to further calls.
        import numpy as np
        from sbm.sbm_simulator import GenerateStochasticBlockModelWithFeatures

        # Parameterize me...
        nvertex_min = 5
        nvertex_max = 50
        nedges_min = 10
        nedges_max = 100
        edge_center_distance_min = 1.0
        edge_center_distance_max = 10.0

        num_vertices = np.random.randint(nvertex_min, nvertex_max+1)
        num_edges = np.random.randint(nedges_min, nedges_max+1)
        feature_dim = 16
        edge_center_distance = 2.0
        edge_feature_dim = 4

        generator_config = {
            'generator_name': 'StochasticBlockModel',
            'num_verticies': num_vertices,
            'num_edges': num_edges,
            'feature_dim': feature_dim,
            'feature_center_distance': 2.0,
            'edge_center_distance': edge_center_distance,
            'edge_feature_dim': 4
        }

        data = GenerateStochasticBlockModelWithFeatures(
            num_vertices = generator_config['num_verticies'],
            num_edges = generator_config['num_edges'],
            pi = np.array([0.25, 0.25, 0.25, 0.25]),
            prop_mat = np.ones((4, 4)) + 9.0 * np.diag([1,1,1,1]),
            feature_center_distance = generator_config['feature_center_distance'],
            feature_dim=generator_config['feature_dim'],
            edge_center_distance = generator_config['edge_center_distance'],
            edge_feature_dim = generator_config['edge_feature_dim']
        )

        yield {'sample_id': sample_id,
               'generator_config': generator_config,
               'data': data}

class WriteSbmDoFn(beam.DoFn):

    def __init__(self, output_path):
        self.output_path_ = output_path

    def process(self, element):
        import apache_beam
        import numpy as np

        sample_id = element['sample_id']
        config = element['generator_config']
        data = element['data']

        text_mime = 'text/plain'
        prefix = '{0:05}'.format(sample_id)
        config_object_name = os.path.join(self.output_path_, prefix + '_config.txt')
        with apache_beam.io.filesystems.FileSystems.create(config_object_name, text_mime) as f:
            buf = bytes(json.dumps(config), 'utf-8')
            f.write(buf)
            f.close()

        graph_object_name = os.path.join(self.output_path_, prefix + '_graph.gt')
        with apache_beam.io.filesystems.FileSystems.create(graph_object_name) as f:
            data.graph.save(f)
            f.close()

        graph_memberships_object_name = os.path.join(
            self.output_path_, prefix + '_graph_memberships.txt')
        with apache_beam.io.filesystems.FileSystems.create(graph_memberships_object_name, text_mime) as f:
            np.savetxt(f, data.graph_memberships)
            f.close()


        node_features_object_name = os.path.join(
            self.output_path_, prefix + '_node_features.txt')
        with apache_beam.io.filesystems.FileSystems.create(node_features_object_name, text_mime) as f:
            np.savetxt(f, data.node_features)
            f.close()

        feature_memberships_object_name = os.path.join(
            self.output_path_, prefix + '_feature_membership.txt')
        with apache_beam.io.filesystems.FileSystems.create(feature_memberships_object_name, text_mime) as f:
            np.savetxt(f, data.feature_memberships)
            f.close()

        edge_features_object_name = os.path.join(
            self.output_path_, prefix + '_edge_features.txt')
        with apache_beam.io.filesystems.FileSystems.create(edge_features_object_name, text_mime) as f:
            for edge_tuple, features in data.edge_features.items():
                buf = bytes('{0},{1},{2}'.format(edge_tuple[0], edge_tuple[1], features), 'utf-8')
                f.write(buf)
            f.close()


class ConvertToTorchGeoDataParDo(beam.DoFn):
    def process(self, element):
        sample_id = element['sample_id']
        data = element['data']
        from sbm.utils import sbm_data_to_torchgeo_data
        yield sample_id, sbm_data_to_torchgeo_data(data)


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

    args, pipeline_args = parser.parse_known_args(argv)

    logging.info(f'output: {args.output}')
    logging.info(f'Pipeline Aargs: {pipeline_args}')
    print(f'Pipeline Args: {pipeline_args}')

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = False

    with beam.Pipeline(options=pipeline_options) as p:

        graph_samples = (
            p
            | 'Create Sample Ids' >> beam.Create(range(args.nsamples))
            | 'Sample Graphs' >> beam.ParDo(SampleSbmDoFn())
        )

        (graph_samples | 'Write Sampled Graph' >> beam.ParDo(WriteSbmDoFn(args.output)))

        torchgeo_data = (graph_samples | 'Convert to torchgeo data.' >> beam.ParDo(ConvertToTorchGeoDataParDo()))


if __name__ == '__main__':
    main(None)
