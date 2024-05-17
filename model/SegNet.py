import torch
from torch import nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        conv = self.conv(x)
        return self.pool(conv), conv


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetAll(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_blocks = n_blocks

        self.down_blocks = nn.ModuleList(
            [Down(n_channels * 2 ** i, n_channels * 2 ** (i + 1)) for i in range(n_blocks)])

        self.up_blocks = nn.ModuleList(
            [Up(n_channels * 2 ** (i + 2), n_channels * 2 ** (i + 1)) for i in range(n_blocks - 1, -1, -1)])

        self.bottle_neck = nn.Sequential(
            nn.Conv1d(n_channels * 2 ** n_blocks, n_channels * 2 ** (n_blocks + 1), 3, padding=1),
            nn.BatchNorm1d(n_channels * 2 ** (n_blocks + 1)),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_channels * 2 ** (n_blocks + 1), n_channels * 2 ** (n_blocks + 1), 3, padding=1),
            nn.BatchNorm1d(n_channels * 2 ** (n_blocks + 1)),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv1d(n_channels * 2, n_channels * n_classes, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        skip_x = []
        for i in range(self.n_blocks):
            x, skip = self.down_blocks[i](x)
            skip_x.append(skip)
        x = self.bottle_neck(x)
        for i in range(self.n_blocks):
            x = self.up_blocks[i](x, skip_x[-i - 1])
        x = self.out_conv(x)
        x = self.final_act(x)
        b, c, l = x.shape
        x = x.view(b, self.n_channels, self.n_classes, l)
        x = x.permute(0, 1, 3, 2)
        return x


class UNetSingle(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_blocks = n_blocks

        self.conv_head = nn.Sequential(
            nn.Conv1d(1, n_channels, 3, padding=1),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [Down(n_channels * 2 ** i, n_channels * 2 ** (i + 1)) for i in range(n_blocks)])

        self.up_blocks = nn.ModuleList(
            [Up(n_channels * 2 ** (i + 2), n_channels * 2 ** (i + 1)) for i in range(n_blocks - 1, -1, -1)])

        self.bottle_neck = nn.Sequential(
            nn.Conv1d(n_channels * 2 ** n_blocks, n_channels * 2 ** (n_blocks + 1), 3, padding=1),
            nn.BatchNorm1d(n_channels * 2 ** (n_blocks + 1)),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_channels * 2 ** (n_blocks + 1), n_channels * 2 ** (n_blocks + 1), 3, padding=1),
            nn.BatchNorm1d(n_channels * 2 ** (n_blocks + 1)),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv1d(n_channels * 2, n_classes, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), 1, -1)
        x = self.conv_head(x)
        skip_x = []
        for i in range(self.n_blocks):
            x, skip = self.down_blocks[i](x)
            skip_x.append(skip)
        x = self.bottle_neck(x)
        for i in range(self.n_blocks):
            x = self.up_blocks[i](x, skip_x[-i - 1])
        x = self.out_conv(x)
        x = self.final_act(x)
        b, c, l = x.shape
        x = x.view(-1, self.n_channels, self.n_classes, l)
        x = x.permute(0, 1, 3, 2)
        return x



if __name__ == '__main__':

    model = UNetAll(n_channels=22, n_classes=1, n_blocks=5)
    x = torch.randn(5, 22, 625)
    print(model(x).shape)
    pass