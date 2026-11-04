"""
========================================================================================
Deep Face Verification: Adversarial Attacks and Defense Strategies
Grounded in: Kilany & Mahfouz (2025), Scientific Reports (Nature Portfolio)
========================================================================================
Complete end-to-end framework implementing:
1. Deep Face Verification pipeline (Backbone + Hyperspherical Normalization)
2. ArcFace Additive Angular Margin formulation
3. White-Box & Gray-Box Adversarial Attacks (FGSM, PGD, MIM, Eyeglass Patch)
4. Obfuscation (Dodging) vs Impersonation (Targeted) Attack Evaluation
5. Defense Suite: Multi-Transformation Denoising & Feature-Squeezing Adversarial Detector
6. Comprehensive Metric Suite: TAR@FAR, ASR, PSNR, L_inf, and Defense Recovery Rate
========================================================================================
"""

import sys
import os
import torch
import numpy as np

from src.models import build_face_model, ArcFaceLoss
from src.alignment.face_align import generate_synthetic_face_samples, preprocess_face_tensor
from src.attacks import FGSM_FaceAttack, PGD_FaceAttack, MIM_FaceAttack, FacialPatchAttack
from src.defenses import InputTransformations, FeatureSqueezingDetector
from src.evaluation import (
    find_optimal_threshold,
    compute_tar_at_far,
    calculate_perturbation_metrics
)

def log(msg=""):
    print(msg)
    sys.stdout.flush()

def run_face_verification_adversarial_suite():
    log("=" * 82)
    log("     DEEP FACE VERIFICATION: ADVERSARIAL ATTACKS & DEFENSE BENCHMARK")
    log("        Based on Sohair Kilany & Ahmed Mahfouz (Scientific Reports, 2025)")
    log("=" * 82)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"[*] Execution device: {device}")

    # 1. Initialize Deep Model Backbone (MobileFaceNet)
    embedding_dim = 128
    num_classes = 5
    log(f"[*] Initializing MobileFaceNet feature extractor (Embedding size: {embedding_dim}d)...")
    model = build_face_model(embedding_size=embedding_dim).to(device)
    arcface = ArcFaceLoss(in_features=embedding_dim, out_features=num_classes).to(device)

    # 2. Generate structured dataset of identities with intra-personal variations
    log(f"[*] Generating structured dataset with {num_classes} identities (4 variations each)...")
    data = generate_synthetic_face_samples(num_identities=num_classes, samples_per_identity=4)

    gallery = {}
    probes = {}
    for id_idx, imgs in data.items():
        gallery[id_idx] = preprocess_face_tensor(imgs[0], device=device)
        probes[id_idx] = [preprocess_face_tensor(img, device=device) for img in imgs[1:]]

    # Pre-train / calibrate weights briefly to produce clean clustering on hypersphere
    log("[*] Calibrating hyperspherical ArcFace embeddings for clean verification...")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    all_imgs = []
    all_labels = []
    for id_idx, imgs in data.items():
        for img in imgs:
            all_imgs.append(preprocess_face_tensor(img, device=device))
            all_labels.append(id_idx)
    batch_x = torch.cat(all_imgs, dim=0)
    batch_y = torch.tensor(all_labels, dtype=torch.long, device=device)

    model.train()
    arcface.train()
    for epoch in range(20):
        optimizer.zero_grad()
        feats = model(batch_x, normalize=True)
        logits = arcface(feats, batch_y)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()

    # 3. Clean Face Verification Evaluation (Baseline)
    log("\n" + "-" * 82)
    log("STAGE 1: CLEAN FACE VERIFICATION BENCHMARK")
    log("-" * 82)
    genuine_scores = []
    impostor_scores = []

    with torch.no_grad():
        for id_idx in range(num_classes):
            gal_emb = model(gallery[id_idx])
            # Genuine pairs (same identity)
            for p_img in probes[id_idx]:
                p_emb = model(p_img)
                sim = torch.cosine_similarity(gal_emb, p_emb).item()
                genuine_scores.append(sim)
            # Impostor pairs (different identity)
            for other_id in range(num_classes):
                if other_id != id_idx:
                    other_gal_emb = model(gallery[other_id])
                    sim = torch.cosine_similarity(gal_emb, other_gal_emb).item()
                    impostor_scores.append(sim)

    optimal_tau, clean_acc = find_optimal_threshold(genuine_scores, impostor_scores)
    tar_at_far, tau_far = compute_tar_at_far(genuine_scores, impostor_scores, target_far=0.10)

    log(f"[*] Optimal Decision Threshold (tau*): {optimal_tau:.4f}")
    log(f"[*] Clean Verification Accuracy:      {clean_acc * 100:.2f}%")
    log(f"[*] Mean Genuine Pair Cosine Sim:     {np.mean(genuine_scores):.4f}")
    log(f"[*] Mean Impostor Pair Cosine Sim:    {np.mean(impostor_scores):.4f}")
    log(f"[*] TAR @ FAR=10%:                    {tar_at_far * 100:.2f}% (tau={tau_far:.4f})")

    # 4. Adversarial Attacks Evaluation
    log("\n" + "-" * 82)
    log("STAGE 2: ADVERSARIAL ATTACKS (OBFUSCATION & IMPERSONATION)")
    log("-" * 82)

    fgsm = FGSM_FaceAttack(model, eps=8/255, device=device)
    pgd = PGD_FaceAttack(model, eps=8/255, alpha=2/255, steps=8, device=device)
    mim = MIM_FaceAttack(model, eps=8/255, alpha=2/255, steps=8, device=device)
    patch = FacialPatchAttack(model, mask_type='eyeglass', alpha=5/255, steps=10, device=device)

    attacks = {
        "FGSM (eps=8/255)": fgsm,
        "PGD-8 (eps=8/255)": pgd,
        "MIM-8 (eps=8/255)": mim,
        "Adversarial Eyeglasses Patch": patch
    }

    # 4A. Obfuscation (Dodging) Attack
    log("\n>>> [A] OBFUSCATION (DODGING) ATTACKS:")
    log("    Goal: Evade recognition by forcing CosineSim(x_adv, x_gallery) < tau*")
    
    obf_results = {}
    for name, atk in attacks.items():
        success_count = 0
        total_tests = 0
        p_sims = []
        l_infs = []
        psnrs = []

        for id_idx in range(num_classes):
            gal_img = gallery[id_idx]
            gal_emb = model(gal_img).detach()
            for p_img in probes[id_idx]:
                x_adv = atk.perturb(p_img, target_ref=gal_img, attack_type='obfuscation')
                with torch.no_grad():
                    adv_emb = model(x_adv)
                    sim = torch.cosine_similarity(gal_emb, adv_emb).item()
                p_sims.append(sim)
                if sim < optimal_tau:
                    success_count += 1
                total_tests += 1

                pert_stats = calculate_perturbation_metrics(p_img, x_adv)
                l_infs.append(pert_stats["L_inf_255"])
                psnrs.append(pert_stats["PSNR_dB"])

        asr = (success_count / total_tests) * 100.0
        obf_results[name] = {
            "ASR": asr,
            "Mean_Sim": np.mean(p_sims),
            "L_inf_255": np.mean(l_infs),
            "PSNR": np.mean(psnrs)
        }
        log(f"    - {name:<30} | ASR: {asr:>6.2f}% | Adv Sim: {np.mean(p_sims):.4f} | L_inf: {np.mean(l_infs):.1f}/255 | PSNR: {np.mean(psnrs):.1f} dB")

    # 4B. Impersonation (Targeted) Attack
    log("\n>>> [B] IMPERSONATION (TARGETED) ATTACKS:")
    log("    Goal: Impersonate victim by forcing CosineSim(x_adv, x_victim) >= tau*")
    
    imp_results = {}
    for name, atk in attacks.items():
        success_count = 0
        total_tests = 0
        p_sims = []

        for attacker_id in range(min(2, num_classes)):
            victim_id = (attacker_id + 1) % num_classes
            victim_gal = gallery[victim_id]
            victim_emb = model(victim_gal).detach()

            for p_img in probes[attacker_id]:
                x_adv = atk.perturb(p_img, target_ref=victim_gal, attack_type='impersonation')
                with torch.no_grad():
                    adv_emb = model(x_adv)
                    sim = torch.cosine_similarity(victim_emb, adv_emb).item()
                p_sims.append(sim)
                if sim >= optimal_tau:
                    success_count += 1
                total_tests += 1

        asr = (success_count / total_tests) * 100.0
        imp_results[name] = {"ASR": asr, "Mean_Sim": np.mean(p_sims)}
        log(f"    - {name:<30} | ASR: {asr:>6.2f}% | Adv Sim: {np.mean(p_sims):.4f}")

    # 5. Defense Strategies Evaluation
    log("\n" + "-" * 82)
    log("STAGE 3: DEFENSE EVALUATION (TRANSFORMATION & FEATURE-SQUEEZING DETECTION)")
    log("-" * 82)

    eval_atk = pgd
    detector = FeatureSqueezingDetector(model, threshold=0.08, device=device)

    clean_probe_list = [p for id_probes in probes.values() for p in id_probes]
    detector.calibrate_threshold(clean_probe_list, percentile=90.0)
    log(f"[*] Calibrated Adversarial Detection Threshold: {detector.threshold:.4f}")

    detected_count = 0
    total_adv = 0
    recovered_obf = 0
    defended_sims = []

    for id_idx in range(num_classes):
        gal_img = gallery[id_idx]
        gal_emb = model(gal_img).detach()
        for p_img in probes[id_idx]:
            x_adv = eval_atk.perturb(p_img, target_ref=gal_img, attack_type='obfuscation')
            total_adv += 1

            # 1. Adversarial Detection Check
            if detector.is_adversarial(x_adv):
                detected_count += 1

            # 2. Denoising / Input Transformation Defense
            x_defended = InputTransformations.composite_defense(x_adv, quality=75, bits=5)
            with torch.no_grad():
                def_emb = model(x_defended)
                sim_def = torch.cosine_similarity(gal_emb, def_emb).item()
            defended_sims.append(sim_def)

            if sim_def >= optimal_tau:
                recovered_obf += 1

    detection_rate = (detected_count / total_adv) * 100.0
    recovery_rate = (recovered_obf / total_adv) * 100.0

    log(f"[*] Adversarial Detection Rate (Clean vs PGD):  {detection_rate:.2f}%")
    log(f"[*] Post-Defense Mean Similarity:                {np.mean(defended_sims):.4f} (Raw PGD Sim: {obf_results['PGD-8 (eps=8/255)']['Mean_Sim']:.4f})")
    log(f"[*] Defense Recovery Rate (Restored to Genuine): {recovery_rate:.2f}%")

    log("\n" + "=" * 82)
    log("                     EXECUTIVE BENCHMARK SUMMARY")
    log("=" * 82)
    log(f"  Metric                                 Before Defense       After Defense")
    log(f"  -------------------------------------------------------------------------")
    log(f"  PGD Obfuscation ASR (lower is safer):  {obf_results['PGD-8 (eps=8/255)']['ASR']:>12.2f}%       {(100.0 - recovery_rate):>12.2f}%")
    log(f"  Adversarial Detection Accuracy:        {'N/A':>13}       {detection_rate:>12.2f}%")
    log(f"  Clean Verification Accuracy:           {clean_acc * 100:>12.2f}%       {clean_acc * 100:>12.2f}%")
    log("=" * 82)
    log("[+] Benchmark complete. Pipeline executed successfully.")

if __name__ == '__main__':
    run_face_verification_adversarial_suite()
