import torch
import torch.nn as nn
import torch.nn.functional as F

# Triplet Attention Module (from your provided code)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = (x_out + x_out11 + x_out21) / 3
        else:
            x_out = (x_out11 + x_out21) / 2
        return x_out

# Your seismic velocity prediction model
class SeismicVelocityNet(nn.Module):
    def __init__(self):
        super(SeismicVelocityNet, self).__init__()

        # Initial conv layers to extract features from (5, 1000, 70)
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.attn1 = TripletAttention()

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # Downsample height and width approx by 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.attn2 = TripletAttention()

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # Further downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attn3 = TripletAttention()

        # After these conv + attention layers, 
        # height: 1000 -> ~250 (with stride 2 twice)
        # width: 70 -> ~17 or 18

        # We need to reduce height and width further to match output size 70x70
        # So let's use adaptive pooling to get (128, 70, 70)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((70, 70))

        # Final conv layer to produce single channel velocity map output
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        """
        x shape: (batch, 5, 1000, 70)
        output shape: (batch, 70, 70)
        """
        x = self.conv1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = self.attn3(x)

        # Adaptive pool to get desired output spatial shape
        x = self.adaptive_pool(x)  # (batch, 128, 70, 70)

        x = self.final_conv(x)     # (batch, 1, 70, 70)
        x = x.squeeze(1)           # (batch, 70, 70)

        return x

# Test run
if __name__ == "__main__":
    model = SeismicVelocityNet()
    input_tensor = torch.randn(6000, 5, 1000, 70)  # example batch of 6000 (may use smaller batch in practice)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected: (6000, 70, 70)
