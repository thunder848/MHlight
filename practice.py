import torch
import math
import os
import pickle
import random
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_edge_index(pre_adj, num_neighoor):

    adj = np.array([])

    pre_adj = np.array(pre_adj)
    send_node = pre_adj[:,0].reshape(-1,1)

    for i in range(1, num_neighoor):
        receive_node = pre_adj[:, i].reshape(-1,1)
        temp_edge = np.hstack((send_node , receive_node))
        if i == 1:
            adj = temp_edge
        else:
            adj = np.concatenate((adj,temp_edge))
    return adj


if __name__ == '__main__':
    pass
    pre_adj = [[0,1,2,3,4],
               [1,2,3,4,5],
               [2,3,4,5,6]]
    num_neighoor = 5
    adj = get_edge_index(pre_adj,num_neighoor)
    print(adj)

