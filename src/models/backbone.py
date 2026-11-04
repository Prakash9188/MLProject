import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion=2):
        super(DepthwiseSeparable, self).__init__()
        hidden_dim = in_c * expansion
        self.use_residual = (stride == 1 and in_c == out_c)
        
        self.block = nn.Sequential(
            ConvBlock(in_c, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            LinearBlock(hidden_dim, out_c, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet Backbone: Efficient deep architecture optimized for face verification.
    Outputs normalized 128-d or 512-d feature embeddings.
    """
    def __init__(self, embedding_size: int = 128):
        super(MobileFaceNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.dw1 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, groups=64)

        self.blocks = nn.Sequential(
            DepthwiseSeparable(64, 64, stride=2, expansion=2),
            DepthwiseSeparable(64, 64, stride=1, expansion=2),
            DepthwiseSeparable(64, 128, stride=2, expansion=2),
            DepthwiseSeparable(128, 128, stride=1, expansion=2),
            DepthwiseSeparable(128, 128, stride=2, expansion=2),
            DepthwiseSeparable(128, 128, stride=1, expansion=2),
        )

        self.conv2 = ConvBlock(128, 256, kernel_size=1, stride=1, padding=0)
        # Global Depthwise Convolution (GDConv) replacing Global Average Pooling
        self.gdconv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dw1(out)
        out = self.blocks(out)
        out = self.conv2(out)
        embeddings = self.gdconv(out)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

def build_face_model(embedding_size: int = 128) -> MobileFaceNet:
    model = MobileFaceNet(embedding_size=embedding_size)
    return model
