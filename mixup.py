import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.data import Data
from graph_conv import GraphConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

import pdb
import numpy as np
import random
import copy
import argparse

parser = argparse.ArgumentParser('Mixup')
parser.add_argument('--mixup', action='store_true', help='Whether to have Mixup')
args = parser.parse_args()

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index, edge_index_b, lam, id_new_value_old):

        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)
        
        x0_b = x0[id_new_value_old]
        x1_b = x1[id_new_value_old]
        x2_b = x2[id_new_value_old]

        x0_mix = x0 * lam + x0_b * (1 - lam)

        new_x1 = self.conv1(x0, edge_index, x0_mix)
        new_x1_b = self.conv1(x0_b, edge_index_b, x0_mix)
        new_x1 = F.relu(new_x1)
        new_x1_b = F.relu(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=0.4, training=self.training)

        new_x2 = self.conv2(x1, edge_index, x1_mix)
        new_x2_b = self.conv2(x1_b, edge_index_b, x1_mix)
        new_x2 = F.relu(new_x2)
        new_x2_b = F.relu(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=0.4, training=self.training)

        new_x3 = self.conv3(x2, edge_index, x2_mix)
        new_x3_b = self.conv3(x2_b, edge_index_b, x2_mix)
        new_x3 = F.relu(new_x3)
        new_x3_b = F.relu(new_x3_b)

        x3_mix = new_x3 * lam + new_x3_b * (1 - lam)
        x3_mix = F.dropout(x3_mix, p=0.4, training=self.training)

        x = x3_mix
        x = self.lin(x)
        return x.log_softmax(dim=-1)


# set random seed
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.


# load data
dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


# split data
node_id = np.arange(data.num_nodes)
np.random.shuffle(node_id)
data.train_id = node_id[:int(data.num_nodes * 0.6)]
data.val_id = node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)]
data.test_id = node_id[int(data.num_nodes * 0.8):]


# define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=256, in_channel = dataset.num_node_features, out_channel = dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# func train one epoch
def train(data):
    model.train()

    if args.mixup:
        lam = np.random.beta(4.0, 4.0)
    else:
        lam = 1.0

    data_b, id_new_value_old = shuffleData(data)
    data = data.to(device)
    data_b = data_b.to(device)

    optimizer.zero_grad()

    out = model(data.x, data.edge_index, data_b.edge_index, lam, id_new_value_old)
    loss = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam + \
           F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam)

    loss.backward()
    optimizer.step()

    return loss.item()


# test
@torch.no_grad()
def test(data):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device), data.edge_index.to(device), 1, np.arange(data.num_nodes))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, id_ in data('train_id', 'val_id', 'test_id'):
        accs.append(correct[id_].sum().item() / id_.shape[0])
    return accs


best_acc = 0
accord_epoch = 0
accord_train_acc = 0
accord_train_loss = 0
for epoch in range(1, 300):
    loss = train(data)
    accs = test(data)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Test Acc: {accs[2]:.4f}')