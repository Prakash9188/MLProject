import cv2
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO
from PIL import Image

class InputTransformations:
    """
    Collection of Input Transformation Defenses for Deep Face Verification.
    Surveyed in Kilany & Mahfouz (2025) Section 5:
    - Bit-depth reduction (Color Quantization)
    - Spatial Smoothing (Gaussian & Median filtering)
    - Total Variation Minimization (TVM)
    - JPEG Compression Artifact Simulation
    """
    @staticmethod
    def bit_depth_reduction(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        Reduces color depth to 2^bits quantization levels.
        Squeezes out subtle, imperceptible gradient perturbations.
        """
        levels = 2 ** bits
        step = 1.0 / (levels - 1)
        quantized = torch.round(x / step) * step
        return torch.clamp(quantized, 0.0, 1.0)

    @staticmethod
    def gaussian_blur(x: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
        """Applies 2D Gaussian blur filter on tensor."""
        channels = x.shape[1]
        k = kernel_size
        ax = np.arange(-k // 2 + 1., k // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        kernel_tensor = torch.from_numpy(kernel).float().repeat(channels, 1, 1, 1).to(x.device)
        pad = k // 2
        blurred = F.conv2d(x, kernel_tensor, padding=pad, groups=channels)
        return torch.clamp(blurred, 0.0, 1.0)

    @staticmethod
    def median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Applies median filtering using OpenCV per channel."""
        device = x.device
        b, c, h, w = x.shape
        x_np = (x.detach().cpu().numpy() * 255.0).astype(np.uint8)
        filtered = np.zeros_like(x_np)
        for i in range(b):
            for ch in range(c):
                filtered[i, ch] = cv2.medianBlur(x_np[i, ch], kernel_size)
        res = torch.from_numpy(filtered.astype(np.float32) / 255.0).to(device)
        return res

    @staticmethod
    def jpeg_compression(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
        """Simulates lossy JPEG compression using PIL in-memory buffer."""
        device = x.device
        b, c, h, w = x.shape
        out_batch = []
        for i in range(b):
            img_np = (x[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buf = BytesIO()
            pil_img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            rec_img = Image.open(buf)
            rec_np = np.array(rec_img).astype(np.float32) / 255.0
            out_batch.append(torch.from_numpy(rec_np).permute(2, 0, 1).unsqueeze(0))
        return torch.cat(out_batch, dim=0).to(device)

    @classmethod
    def composite_defense(cls, x: torch.Tensor, quality: int = 80, bits: int = 5) -> torch.Tensor:
        """Combines JPEG compression + Bit-depth reduction for robust multi-defense."""
        x_def = cls.jpeg_compression(x, quality=quality)
        x_def = cls.bit_depth_reduction(x_def, bits=bits)
        return x_def
