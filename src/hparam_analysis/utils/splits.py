import collections
import random

import numpy as np

def get_random_split(data, random_seed=12345):
  y = data.y.numpy()
  labelset = set(y)
  train_mask = data.train_mask.numpy()
  val_mask = data.val_mask.numpy()
  test_mask = data.test_mask.numpy()

  clusters = {}
  for cluster_index in labelset:
    clusters[cluster_index] = np.where(y == cluster_index)[0]

  train_groups = y[train_mask]
  train_counts = collections.Counter(train_groups)
  val_groups = y[val_mask]
  val_counts = collections.Counter(val_groups)
  test_groups = y[test_mask]
  test_counts = collections.Counter(test_groups)

  random.seed(random_seed)
  rtrain_mask = np.array([False] * len(y))
  rval_mask = np.array([False] * len(y))
  rtest_mask = np.array([False] * len(y))
  for label in labelset:
    train_ind = [0] * train_counts[label]
    val_ind = [1] * val_counts[label]
    test_ind = [2] * test_counts[label]
    inds = np.array(train_ind + val_ind + test_ind)
    random.shuffle(inds)
    train_indx = list(clusters[label][np.squeeze(np.argwhere(inds == 0))])
    val_indx = list(clusters[label][np.squeeze(np.argwhere(inds == 1))])
    test_indx = list(clusters[label][np.squeeze(np.argwhere(inds == 2))])
    rtrain_mask[train_indx] = True
    rval_mask[val_indx] = True
    rtest_mask[test_indx] = True
  return rtrain_mask, rval_mask, rtest_mask