import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., CVPR 2019).
    Cited as the state-of-the-art loss function in Kilany & Mahfouz (2025).
    Formula:
        L = -log( e^{s * cos(theta_{y_i} + m)} / (e^{s * cos(theta_{y_i} + m)} + sum_{j != y_i} e^{s * cos(theta_j)}) )
    """
    def __init__(self, in_features: int, out_features: int, s: float = 32.0, m: float = 0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weight vector for each class on hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        # 1. Normalize feature vector and weight vector
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        
        if label is None:
            # During inference/matching, return cosine logits directly
            return cosine * self.s

        # 2. Additive angular margin for training: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. One-hot encoding for margin application
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
