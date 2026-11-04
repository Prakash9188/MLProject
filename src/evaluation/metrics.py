import numpy as np
import torch
import torch.nn.functional as F

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D normalized embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))

def find_optimal_threshold(genuine_scores: list, impostor_scores: list):
    """
    Finds optimal decision threshold tau that maximizes verification accuracy.
    genuine_scores: cosine similarities of (same identity) pairs
    impostor_scores: cosine similarities of (different identity) pairs
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    
    thresholds = np.linspace(-1.0, 1.0, 500)
    best_acc = 0.0
    best_tau = 0.5

    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    return best_tau, best_acc

def compute_tar_at_far(genuine_scores: list, impostor_scores: list, target_far: float = 0.01):
    """
    Computes True Accept Rate (TAR) at a specified False Accept Rate (FAR).
    Standard benchmark metric cited in Kilany & Mahfouz (2025).
    """
    sorted_impostor = np.sort(impostor_scores)[::-1]
    far_idx = int(np.floor(target_far * len(sorted_impostor)))
    far_idx = min(max(0, far_idx), len(sorted_impostor) - 1)
    tau = sorted_impostor[far_idx]

    tar = np.mean(np.array(genuine_scores) >= tau)
    return tar, tau

def calculate_perturbation_metrics(x_clean: torch.Tensor, x_adv: torch.Tensor):
    """
    Calculates L_inf, L_2 norm, and PSNR between clean and adversarial image tensors in [0, 1].
    """
    diff = (x_adv - x_clean).detach().cpu().numpy()
    l_inf = float(np.max(np.abs(diff)))
    l_2 = float(np.linalg.norm(diff))
    
    mse = float(np.mean(diff ** 2))
    if mse < 1e-12:
        psnr = 100.0
    else:
        psnr = float(10 * np.log10(1.0 / mse))
        
    return {
        "L_inf": l_inf,
        "L_inf_255": l_inf * 255.0,
        "L_2": l_2,
        "PSNR_dB": psnr
    }
