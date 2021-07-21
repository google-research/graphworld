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

"""Generate random smb configuration files.
"""
import os

from absl import app
from absl import logging
from absl import flags

from dataclasses import dataclass
from typing import Any, Dict, Text

import gin

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('gin_files', None, 'Path to the config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_string('output_path', '/app/data/random_configs', 'Root output directory.')

@gin.configurable
def SampleSmbConfig(
    nsamples,
    nvertex_min=5,
    nvertex_max=50,
    nedges_min=10,
    nedges_max=100,
    feature_dim_min=16,
    feature_dim_max=16,
    edge_center_distance_min=1.0,
    edge_center_distance_max=10.0,
    edge_feature_dim_min=4,
    edge_feature_dim_max=4):

    # Write file locations of configs, one per line.
    output_file_collection_path = os.path.join(FLAGS.output_path, 'configs_collection.txt')
    logging.info(f'nsamples: {nsamples}')
    out_files = []
    for sample in range(nsamples):
        out_file = os.path.join(FLAGS.output_path, "sample_config_{0:05}.gin".format(sample))
        num_vertices = np.random.randint(nvertex_min, nvertex_max+1)
        num_edges = np.random.randint(nedges_min, nedges_max+1)
        feature_dim = np.random.randint(feature_dim_min, feature_dim_max+1)
        edge_center_distance = np.random.uniform(edge_center_distance_min, edge_center_distance_max)
        edge_feature_dim = np.random.randint(edge_feature_dim_min, edge_feature_dim_max + 1)

        generator_config = {
            'num_verticies': num_vertices,
            'num_edges': num_edges,
            'feature_dim': feature_dim,
            'edge_center_distance': edge_center_distance,
            'edge_feature_dim': edge_feature_dim
        }
        logging.info(f'generator_config: {generator_config}')

        logging.info('Writing {}'.format(out_file))
        with open(out_file, 'w') as f:
            f.write("SamplerConfig.generator_name=\"StochasticBlockModel\"\n")
            f.write("SamplerConfig.generator_config={}\n".format(generator_config))

        out_files.append(out_file)

    with open(output_file_collection_path, 'w') as f:
        for fyle in out_files:
            f.write(fyle + '\n')


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    SampleSmbConfig()

if __name__ == '__main__':
    app.run(main)
