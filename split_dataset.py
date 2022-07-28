from nasbench201 import NB201
from nasbench101 import NB101
import math


def get_split_train_set(normal_layers=12, reduction_layers=2, integers2one_hot=True):
    NB101dataset = NB101()
    NB101_X, NB101_y = NB101dataset.load_data_101(integers2one_hot)

    NB201dataset = NB201()
    NB201_X, NB201_y = NB201dataset.load_data_201(integers2one_hot)

    len101 = len(NB101_X)
    len201 = len(NB201_X)
    normal_rate = normal_layers / (normal_layers + reduction_layers)
    # reduction_rate = reduction_layers / (normal_layers + reduction_layers)
    normal101_len = math.floor(len101 * normal_rate)
    normal201_len = math.floor(len201 * normal_rate)

    normal_dataset = (
        NB101_X[:normal101_len] + NB201_X[:normal201_len], NB101_y[:normal101_len] + NB201_y[:normal201_len])
    reduction_dataset = (
        NB101_X[normal101_len:] + NB201_X[normal201_len:], NB101_y[normal101_len:] + NB201_y[normal201_len:])

    return normal_dataset, reduction_dataset