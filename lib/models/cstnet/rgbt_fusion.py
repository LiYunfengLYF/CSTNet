import torch
from torch import nn

from lib.models.cstnet.sfm import SFM
from lib.models.cstnet.cfm import CFM_woGIM, GIM
from lib.utils.token_utils import token2patch, patch2token


class rgbt_layer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.cfm = CFM_woGIM(dim)
        self.gim1 = GIM(dim)
        self.gim2 = GIM(dim)
        self.sfm = SFM(dim, reduction=1, num_heads=4)

    def forward(self, x_v, x_i):
        z_v, x_v = torch.split(x_v, (64, 256,), dim=1)
        z_i, x_i = torch.split(x_i, (64, 256,), dim=1)

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        z_v_res, z_i_res = z_v, z_i
        x_v_res, x_i_res = x_v, x_i

        z_v, z_i = self.cfm(z_v, z_i)
        x_v, x_i = self.cfm(x_v, x_i)

        z_v, z_i = self.gim1(z_v, z_i)
        x_v, x_i = self.gim2(x_v, x_i)

        z_fusion = self.sfm(z_v, z_i)
        x_fusion = self.sfm(x_v, x_i)

        z_v, z_i = z_fusion + z_v_res, z_fusion + z_i_res
        x_v, x_i = x_fusion + x_v_res, x_fusion + x_i_res

        z_v = patch2token(z_v)
        x_v = patch2token(x_v)
        z_i = patch2token(z_i)
        x_i = patch2token(x_i)

        x_v = torch.cat((z_v, x_v), dim=1)
        x_i = torch.cat((z_i, x_i), dim=1)
        return x_v, x_i
