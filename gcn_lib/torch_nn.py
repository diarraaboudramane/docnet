# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d 
import torch_geometric
from torch_geometric.nn import SAGEConv, inits, GCNConv
import torch.nn.functional as F

##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            print(m)
        super(MLP, self).__init__(*m)



class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, drop=0.):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True, groups=4)
        self.batchnorm = norm_layer(norm, out_channels)
        self.activation = act_layer(act)
        self.dropout = nn.Dropout2d(drop)
        
    def forward(self, x):
        # x: Node features (shape: [batch_size, in_channels, num_nodes])
        # adjacency_matrix: Graph adjacency matrix (shape: [batch_size, num_nodes, num_nodes])
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Perform message-passing using adjacency_matrix
        # (details depend on the specific GNN variant)
        return x

# Custom Graph Convolution Layer for document patches
class DFDGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu', norm=None, bias=True):
        super(DFDGraphConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True, groups=4)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        self.act = act_layer(act)
        self.norm = norm_layer(norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = self.act(x)
        if self.norm:
            x = self.norm(x)
        return x
    

class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        print("len(channels) ++++++++++++++++++++++ = ",len(channels))
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
            print(m)
        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                


class BasicSageConv(Seq):
    def __init__(self, channels, act='gelu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(SAGEConv(channels[i - 1], channels[i], 1, bias=bias))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicSageConv, self).__init__(*m)


class Linear(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=bias))
            if norm is not None and norm.lower() != 'none':
                layers.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                # Add activation layer if specified
                if act.lower() == 'relu':
                    layers.append(nn.ReLU())
                elif act.lower() == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise NotImplementedError(f"Activation {act} is not supported.")
            if drop > 0:
                # Add dropout layer if specified
                layers.append(nn.Dropout(drop))
        #layers.append(nn.Linear(channels[-1], 2))
        # Output layer
        super(Linear, self).__init__(*layers)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
