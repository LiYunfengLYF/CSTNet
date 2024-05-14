import math
import torch.nn as nn
from timm.models.layers import trunc_normal_

from lib.models.cstnet.sfm.cfn import CFN
from lib.models.cstnet.sfm.channel_attn import Attention_Module
from lib.models.cstnet.sfm.lpu import LPU


class SFM(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = Attention_Module(dim=dim, reduction=reduction, num_heads=num_heads)
        self.cfn = CFN(in_channels=dim, out_channels=dim)
        self.apply(self._init_weights)

        self.lpu = LPU(dim)

    def forward(self, x1, x2):
        x1 = self.lpu(x1)
        x2 = self.lpu(x2)

        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = x1 + x2
        merge = self.cfn(merge, H, W)

        return merge

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()