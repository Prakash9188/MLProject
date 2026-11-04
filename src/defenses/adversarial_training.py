import torch
import torch.nn as nn
import torch.optim as optim
from ..models.arcface import ArcFaceLoss
from ..attacks.adversarial_attacks import PGD_FaceAttack

class AdversarialFaceTrainer:
    """
    Minimax Adversarial Training Engine for Deep Face Verification.
    Objective:
        min_theta E [ L_ArcFace(f_theta(x_clean), y) + lambda * L_ArcFace(f_theta(x_adv), y) ]
    """
    def __init__(self, model: nn.Module, num_classes: int, embedding_size: int = 128, lr: float = 1e-3, adv_lambda: float = 0.5, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.adv_lambda = adv_lambda
        self.arcface_head = ArcFaceLoss(in_features=embedding_size, out_features=num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.arcface_head.parameters()), lr=lr)
        self.attacker = PGD_FaceAttack(self.model, eps=4/255, alpha=1/255, steps=5, device=device)

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict:
        self.model.train()
        self.arcface_head.train()
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # 1. Clean forward pass
        clean_feats = self.model(x_batch, normalize=True)
        logits_clean = self.arcface_head(clean_feats, y_batch)
        loss_clean = self.criterion(logits_clean, y_batch)

        # 2. Generate adversarial examples
        self.model.eval()
        x_adv_list = []
        for i in range(x_batch.size(0)):
            x_single = x_batch[i:i+1]
            x_adv = self.attacker.perturb(x_single, target_ref=x_single, attack_type='obfuscation')
            x_adv_list.append(x_adv)
        x_adv_batch = torch.cat(x_adv_list, dim=0)

        # 3. Adversarial forward pass
        self.model.train()
        adv_feats = self.model(x_adv_batch, normalize=True)
        logits_adv = self.arcface_head(adv_feats, y_batch)
        loss_adv = self.criterion(logits_adv, y_batch)

        total_loss = loss_clean + self.adv_lambda * loss_adv

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "loss_clean": loss_clean.item(),
            "loss_adv": loss_adv.item()
        }
