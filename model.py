import torch
import torchvision
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1) # For Hin == Hout and Win == Wout
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=0, dropout=0):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.convt2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1) # For Hin == Hout and Win == Wout
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.convt2(x)
        x = self.bn2(x)

        return x


class ColorNet(nn.Module):
    def __init__(self, downchannels=[1, 64, 128, 256], upchannels=[256, 128, 64, 2], skip_connect=False, dropout=0):
        super().__init__()

        self.skip = skip_connect

        self.downblocks = nn.ModuleList([
            DownBlock(downchannels[i], downchannels[i+1], dropout=dropout) for i in range(len(downchannels)-1)
        ])

        self.upblocks = nn.ModuleList([
            UpBlock(upchannels[i], upchannels[i+1], dropout=dropout) for i in range(len(upchannels)-2)
        ])

        self.upblocks.append(UpBlock(upchannels[len(upchannels)-2], upchannels[len(upchannels)-1], output_padding=1, dropout=dropout))

    def forward(self, x):

        if self.skip:
            res = []
        for down in self.downblocks:
            x = down(x)
            if self.skip:
                res.append(x)
        
        if self.skip:
            res.pop()
        for up in self.upblocks:
            x = up(x)
            if self.skip and len(res) != 0:
                x += res.pop()
        
        return x

if __name__ == "__main__":
    model1 = ColorNet(skip_connect=False)
    model2 = ColorNet(skip_connect=True)

    x = torch.rand(1, 1, 128, 128)

    print(f"model1: {model1(x).shape}")
    print(f"model2: {model2(x).shape}")



