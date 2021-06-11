"""Simple test program for generating SMB graph sample.
"""
from absl import app
from absl import logging
from absl import flags

from dataclasses import dataclass
from typing import Any, Dict, Text

import gin

import numpy as np

from graph_tool.all import *

from smb.smb_simulator import GenerateStochasticBlockModelWithFeatures

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("gin_files", None, "Path to the config files.")
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")

@gin.configurable
@dataclass
class SamplerConfig:
    generator_name: str
    generator_config: Dict[Text, Any]

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    sampler_config = SamplerConfig()

    logging.info('sampler_config: %s', sampler_config)

    data = None
    if sampler_config.generator_name == "StochasticBlockModel":
        generator_config = sampler_config.generator_config

        data = GenerateStochasticBlockModelWithFeatures(
            num_vertices = generator_config["num_verticies"],
            num_edges = generator_config["num_edges"],
            pi=np.array([0.25, 0.25, 0.25,0.25]),
            prop_mat=np.ones((4, 4)) + 9.0 * np.diag(np.ones(4)),
            feature_center_distance=2.0,
            feature_dim=16,
            edge_center_distance=2.0,
            edge_feature_dim=4)
    else:
        raise ValueError("Unknown Generator Name: {}".format(sampler_config.generator_name))


    logging.info('%s', data)

if __name__ == '__main__':
    app.run(main)
