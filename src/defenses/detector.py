import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformations import InputTransformations

class FeatureSqueezingDetector:
    """
    Adversarial Perturbation Detector for Face Verification.
    Based on the Feature Squeezing and Discrepancy Principle (Xu et al. / Kilany & Mahfouz Section 5).
    Evaluates angular shift in feature space between raw image and squeezed image:
        Discrepancy = 1 - CosineSimilarity( f(x), f(T(x)) )
    """
    def __init__(self, model: nn.Module, threshold: float = 0.15, device: str = 'cpu'):
        self.model = model
        self.threshold = threshold
        self.device = device
        self.model.eval()

    def compute_discrepancy(self, x: torch.Tensor) -> float:
        x = x.to(self.device)
        with torch.no_grad():
            f_orig = self.model(x)
            # Apply squeezing transformation (Bit-depth + Gaussian smoothing)
            x_squeezed = InputTransformations.bit_depth_reduction(x, bits=4)
            x_squeezed = InputTransformations.gaussian_blur(x_squeezed, kernel_size=3, sigma=0.7)
            f_squeezed = self.model(x_squeezed)

            cos_sim = F.cosine_similarity(f_orig, f_squeezed, dim=-1).item()
            discrepancy = 1.0 - cos_sim
        return discrepancy

    def is_adversarial(self, x: torch.Tensor) -> bool:
        """Returns True if the face image is detected as an adversarial manipulation."""
        disc = self.compute_discrepancy(x)
        return disc > self.threshold

    def calibrate_threshold(self, clean_samples: list, percentile: float = 95.0) -> float:
        """
        Calibrates threshold using a validation set of clean faces to enforce a 
        given True Acceptance / False Alarm rate.
        """
        discs = []
        for img in clean_samples:
            discs.append(self.compute_discrepancy(img))
        self.threshold = float(np.percentile(discs, percentile))
        return self.threshold
