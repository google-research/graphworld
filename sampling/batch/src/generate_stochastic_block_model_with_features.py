"""Simple test program for generating SMB graph sample.
"""
from absl import app
from absl import logging

import numpy as np

from graph_tool.all import *

from smb import smb_simulator

def main(argv):
    smb_data = smb_simulator.GenerateStochasticBlockModelWithFeatures(
        num_vertices=10,
        num_edges=100,
        pi=np.array([0.25, 0.25, 0.25,0.25]),
         prop_mat=np.ones((4, 4)) + 9.0 * np.diag(np.ones(4)),
        feature_center_distance=2.0,
        feature_dim=16,
        edge_center_distance=2.0,
        edge_feature_dim=4)

    logging.info('%s', smb_data)

if __name__ == '__main__':
    app.run(main)
