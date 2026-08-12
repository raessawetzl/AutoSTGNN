import torch
import numpy as np

def adj_to_edge_index(adj_mx):
    adj = torch.FloatTensor(adj_mx)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # (2, E)
    edge_weight = adj[edge_index[0], edge_index[1]]            # (E,)
    return edge_index, edge_weight