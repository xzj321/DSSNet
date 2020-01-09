import torch
import torch.nn as nn
from utils.same_conv import SameConv2d


class SCSE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)



class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = SameConv2d(in_planes, out_planes,
                              kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResInception(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResInception, self).__init__()
        self.in_c = in_c
        branch_c = int(out_c / 4)
        self.branch0 = BasicConv2d(in_c, branch_c, 1, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_c, branch_c, 1, 1),
            BasicConv2d(branch_c, branch_c, 3, 1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_c, branch_c, 1, 1),
            BasicConv2d(branch_c, branch_c, 3, 1),
            BasicConv2d(branch_c, branch_c, 3, 1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_c, branch_c, 1, 1)
        )
        self.scse = SCSE(out_c)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.scse(out)
        out += x
        return out

class SeRes(nn.Module):
    def __init__(self, c, reduction=16):
        super(SeRes, self).__init__()
        self.con1 = BasicConv2d(c, c, 3, 1)
        self.con2 = BasicConv2d(c, c, 3, 1)
        self.scse = SCSE(c, reduction)
    def forward(self, x):
        residual = x
        x = self.con1(x)
        x = self.con2(x)
        x = self.scse(x)
        x += residual
        return x