import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, input_dim_num, num_heads=507):
        super(AttentionModel, self).__init__()
        self.input_dim = input_dim_num
        self.num_heads = num_heads
        self.flatten_dim = input_dim_num
        self.attention = nn.MultiheadAttention(embed_dim=self.flatten_dim, num_heads=num_heads)

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[1], input_shape[0], self.flatten_dim)
        output, _ = self.attention(x, x, x)
        output = output.permute(1, 0, 2)
        output = output.view(*input_shape)

        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, ConvBlock_conv):
        super(ConvBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=ConvBlock_conv[0],
                                 padding=int((ConvBlock_conv[0] - 1) / 2))  # (k -1)/2
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=ConvBlock_conv[1],
                                 padding=int((ConvBlock_conv[1] - 1) / 2))
        self.conv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=ConvBlock_conv[2],
                                 padding=int((ConvBlock_conv[2] - 1) / 2))

        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out1_1 = self.bn(self.conv1_1(x))
        out3_3 = self.bn(self.conv3_3(x))
        out5_5 = self.bn(self.conv5_5(x))
        out1_3_5 = out1_1 + out3_3 + out5_5 + x
        out = out1_3_5

        return out


class DepSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepSepConv, self).__init__()
        self.dep_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                  groups=in_channels)
        self.poi_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dep_conv(x)
        x = self.poi_conv(x)
        return x


class GA_LRCN_DSCN(nn.Module):
    def __init__(self, Dropout, out_channels, Linear, ConvBlock_conv, in_channels=1, input_dim_num=2028):
        super(GA_LRCN_DSCN, self).__init__()
        self.convblock = ConvBlock(in_channels, out_channels, ConvBlock_conv)
        self.attention = AttentionModel(input_dim_num)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.depsepconv = DepSepConv(in_channels, out_channels)

        self.classification1 = nn.Sequential(
            nn.Conv2d(out_channels, 2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.classification2 = nn.Sequential(
            nn.Linear(2020, Linear[0]),
            nn.BatchNorm1d(Linear[0]),
            nn.ReLU(),
            nn.Dropout(Dropout),
            nn.Linear(Linear[0], Linear[1]),
            nn.BatchNorm1d(Linear[1]),
            nn.ReLU(),
            nn.Dropout(Dropout),
            nn.Linear(Linear[1], Linear[2]),
            nn.BatchNorm1d(Linear[2]),
            nn.ReLU(),
            nn.Dropout(Dropout),
            nn.Linear(Linear[2], 2)
        )

    def forward(self, x):
        out1_1 = self.convblock(x)
        out1_2 = self.attention(x)
        out1_1_2 = torch.add(out1_1, out1_2)
        out1_1_2 = self.conv(out1_1_2)
        out2 = self.depsepconv(x)
        out_add = torch.add(out1_1_2, out2)
        out_cl1 = self.classification1(out_add)
        out_cl2 = self.classification2(out_cl1)
        return out_cl2
