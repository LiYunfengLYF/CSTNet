import torch
import torch.nn as nn
from timm.models.layers import Mlp
from torch.nn import init

from lib.models.cstnet.cfm.lsa import LSA
from lib.models.cstnet.cfm.se import SE
from lib.utils.token_utils import patch2token, token2patch


class CFM_woGIM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channels = dim

        self.linear = nn.Linear(dim * 2, self.channels)

        self.lsa = LSA(self.channels, self.channels)

        self.se = SE(self.channels)

        # Initialize linear fusion layers with Kaiming initialization
        init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        x = x.flatten(2).transpose(1, 2).contiguous()

        x_sum = self.linear(x)
        x_sum = self.lsa(x_sum, H, W) + self.se(x_sum, H, W)
        x_fusion = x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x1 + x_fusion, x2 + x_fusion

class GIM(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim * 2, act_layer=act_layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_v, x_i):
        B, C, H, W = x_v.shape
        N = H * W

        x_v = patch2token(x_v)
        x_i = patch2token(x_i)
        x = torch.cat((x_v, x_i), dim=1)

        x = x + self.norm(self.mlp1(x))
        x_v, x_i = torch.split(x, (N, N,), dim=1)
        x_v = token2patch(x_v)
        x_i = token2patch(x_i)

        return x_v, x_i