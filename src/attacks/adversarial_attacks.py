import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdversarialAttack:
    """Base class for Face Verification Adversarial Attacks."""
    def __init__(self, model: nn.Module, eps: float = 8 / 255, device: str = 'cpu'):
        self.model = model
        self.eps = eps
        self.device = device
        self.model.eval()

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between two 1D/2D embedding tensors."""
        return F.cosine_similarity(emb1, emb2, dim=-1)

class FGSM_FaceAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM) adapted for Face Verification.
    Supports both Obfuscation (Dodging) and Impersonation (Targeted) attacks.
    """
    def __init__(self, model: nn.Module, eps: float = 8 / 255, device: str = 'cpu'):
        super(FGSM_FaceAttack, self).__init__(model, eps, device)

    def perturb(self, x: torch.Tensor, target_ref: torch.Tensor, attack_type: str = 'obfuscation') -> torch.Tensor:
        """
        x: Probe image tensor (1, 3, H, W)
        target_ref: Reference image tensor (1, 3, H, W).
                    If 'obfuscation', target_ref is the original clean face x.
                    If 'impersonation', target_ref is the victim/impersonated face.
        """
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        self.model.zero_grad()

        feat_adv = self.model(x_adv)
        with torch.no_grad():
            feat_ref = self.model(target_ref.to(self.device))

        sim = self.cosine_similarity(feat_adv, feat_ref)

        if attack_type == 'obfuscation':
            # Minimize similarity to original face -> gradient ascent on -sim (or descent on sim)
            loss = sim.mean()
            loss.backward()
            grad = x_adv.grad.data
            x_adv = x - self.eps * grad.sign()
        elif attack_type == 'impersonation':
            # Maximize similarity to target face -> gradient ascent on sim
            loss = sim.mean()
            loss.backward()
            grad = x_adv.grad.data
            x_adv = x + self.eps * grad.sign()
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Projection back to valid pixel range [0, 1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        return x_adv

class PGD_FaceAttack(AdversarialAttack):
    """
    Projected Gradient Descent (PGD / BIM) for Face Verification.
    Multi-step iterative attack with L_inf ball projection.
    Cited in Kilany & Mahfouz (2025) as a primary benchmark attack.
    """
    def __init__(self, model: nn.Module, eps: float = 8 / 255, alpha: float = 2 / 255, steps: int = 20, random_start: bool = True, device: str = 'cpu'):
        super(PGD_FaceAttack, self).__init__(model, eps, device)
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def perturb(self, x: torch.Tensor, target_ref: torch.Tensor, attack_type: str = 'obfuscation') -> torch.Tensor:
        x = x.to(self.device)
        target_ref = target_ref.to(self.device)
        
        with torch.no_grad():
            feat_ref = self.model(target_ref)

        # Random initialization within L_inf epsilon ball
        if self.random_start:
            noise = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps).to(self.device)
            x_adv = torch.clamp(x + noise, 0.0, 1.0).detach()
        else:
            x_adv = x.clone().detach()

        for step in range(self.steps):
            x_adv.requires_grad_()
            self.model.zero_grad()
            
            feat_adv = self.model(x_adv)
            sim = self.cosine_similarity(feat_adv, feat_ref).mean()

            if attack_type == 'obfuscation':
                # Minimize similarity: step opposite to gradient of sim
                loss = sim
                loss.backward()
                grad = x_adv.grad.data
                x_adv = x_adv.detach() - self.alpha * grad.sign()
            elif attack_type == 'impersonation':
                # Maximize similarity: step along gradient of sim
                loss = sim
                loss.backward()
                grad = x_adv.grad.data
                x_adv = x_adv.detach() + self.alpha * grad.sign()
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # Project onto L_inf epsilon neighborhood around original x
            eta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()

        return x_adv

class MIM_FaceAttack(AdversarialAttack):
    """
    Momentum Iterative Method (MIM) (Dong et al., CVPR 2018).
    Adds momentum decay to gradients for enhanced transferability in black-box face verification.
    """
    def __init__(self, model: nn.Module, eps: float = 8 / 255, alpha: float = 2 / 255, steps: int = 20, decay: float = 1.0, device: str = 'cpu'):
        super(MIM_FaceAttack, self).__init__(model, eps, device)
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

    def perturb(self, x: torch.Tensor, target_ref: torch.Tensor, attack_type: str = 'obfuscation') -> torch.Tensor:
        x = x.to(self.device)
        target_ref = target_ref.to(self.device)
        x_adv = x.clone().detach()

        with torch.no_grad():
            feat_ref = self.model(target_ref)

        momentum = torch.zeros_like(x)

        for step in range(self.steps):
            x_adv.requires_grad_()
            self.model.zero_grad()
            feat_adv = self.model(x_adv)
            sim = self.cosine_similarity(feat_adv, feat_ref).mean()

            if attack_type == 'obfuscation':
                loss = sim
                loss.backward()
                grad = -x_adv.grad.data  # direction to decrease similarity
            elif attack_type == 'impersonation':
                loss = sim
                loss.backward()
                grad = x_adv.grad.data   # direction to increase similarity
            else:
                raise ValueError(f"Unknown attack: {attack_type}")

            # Normalize gradients by L1 norm
            l1_norm = torch.mean(torch.abs(grad)) + 1e-12
            grad_norm = grad / l1_norm

            # Accumulate velocity/momentum
            momentum = self.decay * momentum + grad_norm
            x_adv = x_adv.detach() + self.alpha * momentum.sign()

            # L_inf projection
            eta = torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()

        return x_adv

class FacialPatchAttack(AdversarialAttack):
    """
    Simulates physical wearable attacks (e.g. Adversarial Eyeglasses / Forehead Sticker).
    Restricts perturbation to a localized binary mask region (e.g. bridge of the nose / glasses frame).
    Sharif et al. (2016) formulation cited in survey Section 4.
    """
    def __init__(self, model: nn.Module, mask_type: str = 'eyeglass', alpha: float = 4 / 255, steps: int = 30, device: str = 'cpu'):
        super(FacialPatchAttack, self).__init__(model, eps=1.0, device=device)
        self.mask_type = mask_type
        self.alpha = alpha
        self.steps = steps

    def get_mask(self, shape, device):
        """Generates a binary mask of eyeglasses/patch on 112x112 aligned face."""
        _, _, H, W = shape
        mask = torch.zeros((1, 1, H, W), device=device)
        if self.mask_type == 'eyeglass':
            # Horizontal glasses bar across eyes (Y: 45 to 58, X: 25 to 87)
            mask[:, :, 46:56, 25:87] = 1.0
            # Left frame
            mask[:, :, 42:62, 30:48] = 1.0
            # Right frame
            mask[:, :, 42:62, 64:82] = 1.0
        elif self.mask_type == 'sticker':
            # Square patch on cheek/forehead
            mask[:, :, 20:40, 45:65] = 1.0
        return mask

    def perturb(self, x: torch.Tensor, target_ref: torch.Tensor, attack_type: str = 'impersonation') -> torch.Tensor:
        x = x.to(self.device)
        target_ref = target_ref.to(self.device)
        mask = self.get_mask(x.shape, self.device)
        
        # Patch starts with random initialization inside mask
        patch_noise = torch.rand_like(x) * mask
        x_adv = (1.0 - mask) * x + mask * patch_noise

        with torch.no_grad():
            feat_ref = self.model(target_ref)

        for _ in range(self.steps):
            x_adv.requires_grad_()
            self.model.zero_grad()
            feat_adv = self.model(x_adv)
            sim = self.cosine_similarity(feat_adv, feat_ref).mean()

            if attack_type == 'obfuscation':
                loss = sim
                loss.backward()
                grad = -x_adv.grad.data
            else:
                loss = sim
                loss.backward()
                grad = x_adv.grad.data

            # Update only pixels within mask
            x_adv = x_adv.detach() + self.alpha * grad.sign() * mask
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            # Ensure pixels outside mask remain unperturbed
            x_adv = (1.0 - mask) * x + mask * x_adv

        return x_adv.detach()
