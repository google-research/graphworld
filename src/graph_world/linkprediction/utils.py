from torch_geometric.utils import train_test_split_edges


def sample_data(data, training_ratio):
  return train_test_split_edges(data, val_ratio=0,
                                test_ratio=1 - training_ratio)
