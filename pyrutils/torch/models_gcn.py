import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class Geo_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        node_n (int): Number of joints in the human body
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
    """

    def __init__(self,
                 node_n,
                 in_channels,
                 out_channels):
        super(Geo_gcn, self).__init__()

        self.joint_embed = embed(in_channels, 64, node_n, norm=True, bias=True)
        self.get_s = compute_similarity(64, 128, bias = True)
        self.weight = Parameter(torch.FloatTensor(64, out_channels))
        self.reset_parameters()
        
    # regularisation
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.joint_embed(x)
        s = self.get_s(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = s.matmul(x)
        x = torch.matmul(x, self.weight)
        x = x.permute(0, 3, 2, 1).contiguous()
        return x

class norm_data(nn.Module):
    def __init__(self, dim= 64, node_n = 19):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim*node_n)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 4, dim1 = 128, node_n = 19, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim, node_n),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1 = 4, dim2 =4, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class compute_similarity(nn.Module):
    def __init__(self, dim1, dim2, bias = False):
        super(compute_similarity, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.s1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.s2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        s1 = self.s1(x).permute(0, 3, 2, 1).contiguous()
        s2 = self.s2(x).permute(0, 3, 1, 2).contiguous()
        s3 = s1.matmul(s2)
        s = self.softmax(s3)
        return s
