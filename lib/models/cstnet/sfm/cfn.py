import torch.nn as nn

class CFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        #
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                                groups=in_channels)
        self.bn33 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                                groups=in_channels)
        self.bn11 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        #
        self.conv_up = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # down
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

        # norm all
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)

        #  + skip-connection
        x = x + self.bn11(self.conv11(x)) + self.bn33(self.conv33(x))

        #  + skip-connection
        x = x + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(x)))))

        x = self.adjust(x)

        out = self.norm(residual + x)
        return out