# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer, Linear, MLP, GraphConvLayer, DFDGraphConvLayer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.layers import DropPath
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import grid




class GraphSAGE2(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        x_k = self.nn2(torch.cat([x, x_j], dim=1))
        return self.sigmoid(x_k) 

class GraphSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphSAGEConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index):
        # Agrégation des caractéristiques des voisins
        row, col = edge_index
        agg_neighbors = torch.zeros_like(x)
        agg_neighbors.index_add_(0, row, x[col])

        # Combinaison des caractéristiques du nœud central et des voisins
        out = torch.cat([x, agg_neighbors], dim=1)
        out = self.linear(out)
        return F.relu(out)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = GraphConvLayer(in_channels, in_channels, act, norm, bias)
        self.nn2 = GraphConvLayer(in_channels*2, out_channels, act, norm, bias)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        
    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x = self.nn1(x)
        x_j, _ = torch.max(x_j, -1, keepdim=True)
        x_k = self.nn2(torch.cat([x, x_j], dim=1))
        return self.sigmoid(x_k) 


# Modified GraphSAGE for Document Forgery Detection
class DFDGraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu', norm=None, bias=True):
        super(DFDGraphSAGE, self).__init__()
        self.nn1 = DFDGraphConvLayer(in_channels, in_channels, act, norm, bias)
        self.nn2 = DFDGraphConvLayer(in_channels * 2, out_channels, act, norm, bias)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification (forged vs. authentic)

    def forward(self, x, edge_index, y=None):
        # x are the features of patches, edge_index defines the edges
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])

        x = self.nn1(x)  # Apply first graph convolution
        x_j, _ = torch.max(x_j, -1, keepdim=True)  # Aggregating neighboring node features
        x_k = self.nn2(torch.cat([x, x_j], dim=1))  # Concatenate and apply second graph convolution
        return self.sigmoid(x_k)  # Output binary classification
    

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        
        if conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'link':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gcn':
            self.gconv = GCN(in_channels,16, out_channels)
        elif conv == 'gnn':
            self.gconv = GNN(in_channels, out_channels)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        
        """self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )"""
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        """B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc3(x)"""
        x = self.drop_path(x) + _tmp
        return x
