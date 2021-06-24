import torch
from torch_geometric.data import Data
import networkx as nx

from torch_geometric.utils import from_networkx

from sbm.sbm_simulator import StochasticBlockModel

def sbm_data_to_torchgeo_data(sbm_data: StochasticBlockModel) -> Dataset:
  nx_graph = nx.Graph()
  edge_tuples = []
  edge_feature_data = []
  for edge in sbm_data.graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])
    ordered_tuple = (edge[0], edge[1])
    if edge[0] > edge[1]:
      ordered_tuple = (edge[1], edge[0])
    edge_feature_data.append(sbm_data.edge_features[ordered_tuple])
    edge_feature_data.append(sbm_data.edge_features[ordered_tuple])

  node_features = torch.tensor(sbm_data.node_features, dtype=torch.float)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  edge_attr = torch.tensor(edge_feature_data, dtype=torch.float)
  labels = torch.tensor(sbm_data.graph_memberships, dtype=torch.long)
  return Data(x=node_features, edge_index=edge_index.t().contiguous(),
              edge_attr=edge_attr, y=labels)
