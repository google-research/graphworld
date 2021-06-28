"""Beam pipeline for generating random graph samples.
"""
import argparse
import os
import json
import setuptools

# from absl import logging

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


class SampleSbmDoFn(beam.DoFn):

    def __init__(self, nvertex_min, nvertex_max, nedges_min, nedges_max):
        self._nvertex_min = nvertex_min
        self._nvertex_max = nvertex_max
        self._nedges_min = nedges_min
        self._nedges_max = nedges_max

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
        edge_center_distance_min = 1.0
        edge_center_distance_max = 10.0
        feature_dim = 16
        edge_center_distance = 2.0
        edge_feature_dim = 4

        num_vertices = np.random.randint(self._nvertex_min, self._nvertex_max)
        num_edges = np.random.randint(self._nedges_min, self._nedges_max)

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
            num_vertices = num_vertices,
            num_edges = num_edges,
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
        self._output_path = output_path

    def process(self, element):
        import apache_beam
        import numpy as np

        sample_id = element['sample_id']
        config = element['generator_config']
        data = element['data']

        # print(f'data.graph_memberships: {data.graph_memberships}')

        text_mime = 'text/plain'
        prefix = '{0:05}'.format(sample_id)
        config_object_name = os.path.join(self._output_path, prefix + '_config.txt')
        with apache_beam.io.filesystems.FileSystems.create(config_object_name, text_mime) as f:
            buf = bytes(json.dumps(config), 'utf-8')
            f.write(buf)
            f.close()

        graph_object_name = os.path.join(self._output_path, prefix + '_graph.gt')
        with apache_beam.io.filesystems.FileSystems.create(graph_object_name) as f:
            data.graph.save(f)
            f.close()

        graph_memberships_object_name = os.path.join(
            self._output_path, prefix + '_graph_memberships.txt')
        with apache_beam.io.filesystems.FileSystems.create(graph_memberships_object_name, text_mime) as f:
            np.savetxt(f, data.graph_memberships)
            f.close()


        node_features_object_name = os.path.join(
            self._output_path, prefix + '_node_features.txt')
        with apache_beam.io.filesystems.FileSystems.create(node_features_object_name, text_mime) as f:
            np.savetxt(f, data.node_features)
            f.close()

        feature_memberships_object_name = os.path.join(
            self._output_path, prefix + '_feature_membership.txt')
        with apache_beam.io.filesystems.FileSystems.create(feature_memberships_object_name, text_mime) as f:
            np.savetxt(f, data.feature_memberships)
            f.close()

        edge_features_object_name = os.path.join(
            self._output_path, prefix + '_edge_features.txt')
        with apache_beam.io.filesystems.FileSystems.create(edge_features_object_name, text_mime) as f:
            for edge_tuple, features in data.edge_features.items():
                buf = bytes('{0},{1},{2}'.format(edge_tuple[0], edge_tuple[1], features), 'utf-8')
                f.write(buf)
            f.close()


class ConvertToTorchGeoDataParDo(beam.DoFn):
    def __init__(self, output_path):
        self._output_path = output_path

    def process(self, element):
        import numpy as np
        import apache_beam
        from sbm.utils import sbm_data_to_torchgeo_data, get_kclass_masks
        sample_id = element['sample_id']
        sbm_data = element['data']

        torchgeo_data = sbm_data_to_torchgeo_data(sbm_data)
        masks = get_kclass_masks(sbm_data)
        torchgeo_stats = {
            'nodes': torchgeo_data.num_nodes,
            'edges': torchgeo_data.num_edges,
            'average_node_degree': torchgeo_data.num_edges / torchgeo_data.num_nodes,
            # 'contains_isolated_nodes': torchgeo_data.contains_isolated_nodes(),
            # 'contains_self_loops': torchgeo_data.contains_self_loops(),
            # 'undirected': bool(torchgeo_data.is_undirected())
        }

        stats_object_name = os.path.join(self._output_path, '{0:05}_torchgeo_stats.txt'.format(sample_id))
        with apache_beam.io.filesystems.FileSystems.create(stats_object_name, 'text/plain') as f:
            buf = bytes(json.dumps(torchgeo_stats), 'utf-8')
            f.write(buf)
            f.close()

        masks_object_name = os.path.join(self._output_path, '{0:05}_masks.txt'.format(sample_id))
        with apache_beam.io.filesystems.FileSystems.create(masks_object_name, 'text/plain') as f:
            for mask in masks:
                np.savetxt(f, np.atleast_2d(mask.numpy()), fmt='%i', delimiter=' ')
            f.close()

        out = {'sample_id': sample_id,
               'torch_data': torchgeo_data,
               'masks': masks}

        yield out


class BenchmarkLinearGCNParDo(beam.DoFn):
    def __init__(self, output_path, num_features, num_classes, hidden_channels, epochs):
        self._output_path = output_path
        self._num_features = num_features
        self._num_classes = num_classes
        self._hidden_channels = hidden_channels
        self._epochs = epochs

    def process(self, element):
        import apache_beam
        from sbm.models import LinearGCN
        sample_id = element['sample_id']
        torch_data = element['torch_data']
        masks = element['masks']

        train_mask, val_mask, test_mask = masks
        linear_model = LinearGCN(
            self._num_features,
            self._num_classes,
            self._hidden_channels,
            train_mask,
            val_mask,
            test_mask)

        losses = linear_model.train(self._epochs, torch_data)
        test_accuracy = linear_model.test(torch_data)

        results = {
            'losses': losses,
            'test_accuracy': test_accuracy
        }

        print(f'results: {results}')

        results_object_name = os.path.join(self._output_path, '{0:05}_results.txt'.format(sample_id))
        with apache_beam.io.filesystems.FileSystems.create(results_object_name, 'text/plain') as f:
            buf = bytes(json.dumps(results), 'utf-8')
            f.write(buf)
            f.close()


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

    args, pipeline_args = parser.parse_known_args(argv)

    print(f'Pipeline Args: {pipeline_args}')

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = False

    # Parameterize
    num_features = 16
    num_classes = 4
    hidden_channels = 8
    epochs = 256

    with beam.Pipeline(options=pipeline_options) as p:

        graph_samples = (
            p
            | 'Create Sample Ids' >> beam.Create(range(args.nsamples))
            | 'Sample Graphs' >> beam.ParDo(
                SampleSbmDoFn(args.nvertex_min, args.nvertex_max,
                              args.nedges_min, args.nedges_max))
        )

        (graph_samples | 'Write Sampled Graph' >> beam.ParDo(WriteSbmDoFn(args.output)))

        (graph_samples
            | 'Convert to torchgeo data.' >> beam.ParDo(ConvertToTorchGeoDataParDo(args.output))
            | 'Benchmark Linear GCN.' >> beam.ParDo(BenchmarkLinearGCNParDo(args.output, num_features, num_classes, hidden_channels, epochs)))


if __name__ == '__main__':
    main(None)
