from torch_geometric.utils import train_test_split_edges


def sample_data(data, training_ratio, tuning_ratio):
  return train_test_split_edges(data, val_ratio=tuning_ratio,
                                test_ratio=1.0 - training_ratio - tuning_ratio)
