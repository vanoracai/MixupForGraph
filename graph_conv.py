import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform
import pdb


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, x_cen):
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(edge_index, size=None, h=h, edge_weight=None)
        return aggr_out + self.lin(x_cen)

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
