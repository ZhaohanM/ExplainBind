# -*- coding: utf-8 -*-
import logging
import os
import io

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

# Project-local imports
from config import parse_config
from features import encode_pretrained_feature
from model import ExplainPLI, EncoderWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, best_model_path, device, args):
    """
    Train with optional attention supervision stored in LMDB.
    Attention supervision is sample-aligned:
        can_maps[i] <-> attn_sites[i]
    """
    best_acc = 0.0
    best_model = None
    epochs_without_improvement = 0

    # ── LMDB for attention supervision ─────────────────────────────
    attn_env = lmdb.open(
        os.path.join(args.data_path, "LMDB", "attention.lmdb"),
        readonly=True,
        lock=False,
    )
    attn_txn = attn_env.begin(buffers=True)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_loss_binary = 0.0
        total_loss_attention = 0.0

        for batch_idx, (prot, prot_mask, drug, drug_mask, attn_sites, labels) in enumerate(train_loader):
            prot = prot.to(device, non_blocking=True)
            drug = drug.to(device, non_blocking=True)
            prot_mask = prot_mask.to(device, non_blocking=True)
            drug_mask = drug_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --------------------------------------------------
            # Unify attn_sites format with test():
            #   List[str], one site per sample
            # --------------------------------------------------
            if isinstance(attn_sites[0], (list, tuple)):
                attn_sites = [s[0] for s in attn_sites]

            assert len(attn_sites) == labels.size(0), \
                f"attn_sites mismatch: {len(attn_sites)} vs batch {labels.size(0)}"

            optimizer.zero_grad()

            # --------------------------------------------------
            # Forward
            # --------------------------------------------------
            output, can_maps, joint_emb = model(prot, prot_mask, drug, drug_mask)

            loss_binary = criterion(output, labels.float())
            loss_att = torch.tensor(0.0, device=device)

            # --------------------------------------------------
            # Attention-guided supervision (optional)
            # --------------------------------------------------
            if args.attention_guided:
                maps = []
                valid_indices = []

                for i, site in enumerate(attn_sites):
                    raw = attn_txn.get(site.encode())
                    if raw is None:
                        if labels[i].item() == 1:
                            print(f"[MISS ATTN SITE] {attn_sites[i]}")
                        continue 

                    amap = torch.tensor(
                        np.load(io.BytesIO(bytes(raw)), allow_pickle=True),
                        dtype=torch.float32,
                        device=device,
                    )

                    # Pad / crop to 512 × 512 × D
                    h, w, d = amap.shape
                    pad = torch.zeros((512, 512, d), device=device)
                    pad[:min(h, 512), :min(w, 512)] = amap[:min(h, 512), :min(w, 512)]

                    maps.append(pad)
                    valid_indices.append(i)

                if maps:
                    # GT attention maps: [N, 512, 512, D]
                    attn_maps_raw = torch.stack(maps)

                    # weighting
                    weights = torch.log1p(attn_maps_raw)

                    # normalise GT over spatial dims
                    sum_spat = attn_maps_raw.sum(dim=(1, 2), keepdim=True)
                    attn_maps = attn_maps_raw / (sum_spat + 1e-8)

                    # predicted attention maps
                    pred_maps = can_maps[valid_indices]
                    pred_sum_spat = pred_maps.sum(dim=(1, 2), keepdim=True)
                    pred_maps = pred_maps / (pred_sum_spat + 1e-8)
                    pred_maps = torch.clamp(pred_maps, min=1e-6)

                    # channel mask (only channels present in GT)
                    ch_mask = (sum_spat.squeeze((1, 2)) > 0).unsqueeze(1).unsqueeze(2).float()

                    kl = F.kl_div(pred_maps.log(), attn_maps, reduction="none")
                    num = (kl * weights * ch_mask).sum((1, 2, 3))
                    den = (weights * ch_mask).sum((1, 2, 3)) + 1e-8
                    loss_att = (num / den).mean()

                loss = (1.0 - args.lambda_attn) * loss_binary + args.lambda_attn * loss_att
            else:
                loss = loss_binary

            # --------------------------------------------------
            # Backward
            # --------------------------------------------------
            loss.backward()

            # --------------------------------------------------
            # FINETUNE CHECK (print once per epoch)
            # --------------------------------------------------
            if batch_idx == 0:
                print("\n========== FINETUNE CHECK ==========")

                print(
                    "Protein encoder requires_grad:",
                    any(p.requires_grad for p in model.prot_encoder.parameters())
                )
                print(
                    "Drug encoder requires_grad:",
                    any(p.requires_grad for p in model.drug_encoder.parameters())
                )

                def _first_grad(module):
                    for name, p in module.named_parameters():
                        if p.requires_grad:
                            return name, p.grad
                    return None, None

                p_name, p_grad = _first_grad(model.prot_encoder)
                d_name, d_grad = _first_grad(model.drug_encoder)

                print("Protein encoder param:", p_name)
                print("  grad is None:", p_grad is None)
                if p_grad is not None:
                    print("  grad norm:", p_grad.norm().item())

                print("Drug encoder param:", d_name)
                print("  grad is None:", d_grad is None)
                if d_grad is not None:
                    print("  grad norm:", d_grad.norm().item())

                prot_ids = {id(p) for p in model.prot_encoder.parameters()}
                drug_ids = {id(p) for p in model.drug_encoder.parameters()}
                opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

                print("Protein encoder in optimiser:", len(prot_ids & opt_ids) > 0)
                print("Drug encoder in optimiser:", len(drug_ids & opt_ids) > 0)

                print("====================================\n")

            optimizer.step()

            total_loss += loss.item()
            total_loss_binary += loss_binary.item()
            total_loss_attention += loss_att.item()

        scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        avg_loss_binary = total_loss_binary / max(1, len(train_loader))
        avg_loss_attention = total_loss_attention / max(1, len(train_loader))

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for _, (prot, prot_mask, drug, drug_mask, attn_sites, labels) in enumerate(valid_loader):
                prot = prot.to(device, non_blocking=True)
                drug = drug.to(device, non_blocking=True)
                prot_mask = prot_mask.to(device, non_blocking=True)
                drug_mask = drug_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if isinstance(attn_sites[0], (list, tuple)):
                    attn_sites = [s[0] for s in attn_sites]

                output, can_maps, joint_emb = model(prot, prot_mask, drug, drug_mask)
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(labels.cpu().numpy())

        auc = roc_auc_score(np.asarray(actuals), np.asarray(predictions))
        val_pred_bin = (np.asarray(predictions) > 0.5).astype(np.int32)
        val_acc = accuracy_score(actuals, val_pred_bin)

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": avg_loss,
            "Loss Binary": avg_loss_binary,
            "Loss Attention": avg_loss_attention,
            "Validation AUC": auc,
            "Validation Accuracy": val_acc,
        })

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Binary: {avg_loss_binary:.4f} | "
            f"Attn: {avg_loss_attention:.4f} | "
            f"Val AUC: {auc:.4f} | Val ACC: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1


        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    attn_env.close()
    return best_model

def test(model, test_loader, device, args):
    """
    GT attention (from LMDB):
        gt_map shape = [ligand_len, protein_len, num_channels]
        - ch 0–4: interaction types
        - ch 5  : unused / empty
        - ch 6  : hydrophobic_interactions
        - ch 7  : overall

    Evaluation (per head / channel):
        - Use the head-specific probability map P[l, r]
        - Select Top-K highest-scoring (ligand token, protein token) pairs
        - Take the unique protein tokens from these Top-K pairs as predictions
        - A hit occurs if any predicted protein token falls within +/- window
          of any GT interacting protein token (existence-based GT residues).
    """

    model.eval()

    # ==================================================
    # Classification metrics containers
    # ==================================================
    all_scores = []
    all_labels = []

    # ==================================================
    # Channel mapping (CRITICAL)
    # ==================================================
    INTERACTION_CHANNELS = {
        "vdw_interaction": 0,
        "hydrogen_bond": 1,
        "salt_bridge": 2,
        "pi_stacking": 3,
        "cation_pi_interaction": 4,
        # ch 5 intentionally unused
        "hydrophobic_interactions": 6,
    }
    INTERACTIONS = list(INTERACTION_CHANNELS.keys())
    OVERALL_CH = 7

    TOPK_LIST = range(1, 16)   # 1..15
    WINDOW_LIST = range(0, 6)  # 0..5
    DEFAULT_VAL = 1e-6

    # ==================================================
    # LMDB for GT attention maps
    # ==================================================
    attn_env = lmdb.open(
        os.path.join(args.data_path, "LMDB", "attention.lmdb"),
        readonly=True,
        lock=False,
    )
    attn_txn = attn_env.begin(buffers=True)

    # ==================================================
    # BPHR stats container
    # ==================================================
    stats = {}
    for name in INTERACTIONS + ["overall"]:
        for K in TOPK_LIST:
            for w in WINDOW_LIST:
                stats[(name, K, w)] = {"hit": 0, "total": 0}

    # ==================================================
    # Helper: Top-K pairwise attention → protein indices
    # ==================================================
    def topk_pairs_to_prot_indices(pair_map: torch.Tensor, K: int) -> torch.Tensor:
        flat = pair_map.reshape(-1)
        k_eff = min(K, flat.numel())
        topk_flat_idx = torch.topk(flat, k=k_eff, largest=True).indices
        prot_idx = (topk_flat_idx % 512).unique()
        return prot_idx

    # ==================================================
    # Evaluation loop
    # ==================================================
    with torch.no_grad():
        for batch_id, (prot, prot_mask, drug, drug_mask, attn_sites, labels) in enumerate(test_loader):
            prot = prot.to(device, non_blocking=True)
            drug = drug.to(device, non_blocking=True)
            prot_mask = prot_mask.to(device, non_blocking=True)
            drug_mask = drug_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if isinstance(attn_sites[0], (list, tuple)):
                attn_sites = [s for sub in attn_sites for s in sub]

            output, can_maps, joint_emb = model(prot, prot_mask, drug, drug_mask)
            # output: [B, 1]
            # can_maps: [B, 512, 512, 8]

            # ------------------------------
            # collect classification outputs
            # ------------------------------
            all_scores.extend(output.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # ------------------------------
            # BPHR evaluation (positives only)
            # ------------------------------
            for i in range(labels.size(0)):
                if labels[i].item() != 1:
                    continue

                site = attn_sites[i]
                raw = attn_txn.get(site.encode())
                if raw is None:
                    continue

                gt_map = torch.tensor(
                    np.load(io.BytesIO(bytes(raw)), allow_pickle=True),
                    dtype=torch.float32,
                    device=device,
                )  # [L_lig, L_prot, C]

                h, w0, c = gt_map.shape
                pad = torch.full((512, 512, c), DEFAULT_VAL, device=device)
                pad[:min(h, 512), :min(w0, 512)] = gt_map[:512, :512]
                gt_map = pad

                pred = can_maps[i]  # [512, 512, 8]
                head_sums = pred.sum(dim=(0, 1), keepdim=True)
                pred_prob = pred / (head_sums + 1e-8)

                # ------------------------------
                # per-interaction BPHR
                # ------------------------------
                for name, ch in INTERACTION_CHANNELS.items():
                    gt_mask = (gt_map[:, :, ch] != DEFAULT_VAL).any(dim=0)
                    gt_idx = torch.nonzero(gt_mask, as_tuple=False).squeeze(-1)
                    if gt_idx.numel() == 0:
                        continue

                    pair_prob = pred_prob[:, :, ch]
                    for K in TOPK_LIST:
                        prot_pred_idx = topk_pairs_to_prot_indices(pair_prob, K)
                        for w in WINDOW_LIST:
                            stats[(name, K, w)]["total"] += 1
                            if torch.any((gt_idx[:, None] - prot_pred_idx[None, :]).abs() <= w):
                                stats[(name, K, w)]["hit"] += 1

                # ------------------------------
                # overall BPHR
                # ------------------------------
                gt_mask = (gt_map[:, :, OVERALL_CH] != DEFAULT_VAL).any(dim=0)
                gt_idx = torch.nonzero(gt_mask, as_tuple=False).squeeze(-1)
                if gt_idx.numel() == 0:
                    continue

                pair_prob = pred_prob[:, :, OVERALL_CH]
                for K in TOPK_LIST:
                    prot_pred_idx = topk_pairs_to_prot_indices(pair_prob, K)
                    for w in WINDOW_LIST:
                        stats[("overall", K, w)]["total"] += 1
                        if torch.any((gt_idx[:, None] - prot_pred_idx[None, :]).abs() <= w):
                            stats[("overall", K, w)]["hit"] += 1

    # ==================================================
    # Save BPHR CSV
    # ==================================================
    records = []
    for (name, K, w), v in stats.items():
        records.append({
            "interaction": name,
            "TopK": K,
            "Window": w,
            "Hits": v["hit"],
            "Total": v["total"],
            "BPHR": v["hit"] / v["total"] if v["total"] > 0 else float("nan"),
        })

    df_bphr = pd.DataFrame(records)
    out_path = os.path.join(args.data_path, f"BPHR_{args.split}.csv")
    df_bphr.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")

    # ==================================================
    # Final classification metrics (Table 1)
    # ==================================================
    y_true = np.asarray(all_labels)
    y_score = np.asarray(all_scores)
    y_pred = (y_score > 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n[Classification Performance]")
    print(f"AUC         : {auc:.4f}")
    print(f"AUPR        : {aupr:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"MCC         : {mcc:.4f}")

    if wandb.run is not None:
        wandb.log({
            "Test/AUC": auc,
            "Test/AUPR": aupr,
            "Test/Accuracy": acc,
            "Test/F1": f1,
            "Test/Sensitivity": sensitivity,
            "Test/Specificity": specificity,
            "Test/MCC": mcc,
        })

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    setup_logging()
    logging.info("Parsing configuration...")
    args = parse_config()

    device = torch.device(args.device)
    logging.info(f"Current device: {args.device}.")

    # Weights & Biases
    wandb.init(
        project="Attention-guided_Full_finetune_PLI_Prediction",
        config=vars(args),
        save_code=True
    )

    # Output directories
    best_model_dir = f"{args.save_path_prefix}{args.dataset}_lambda_{args.lambda_attn}_{args.split}"
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "best_model.ckpt")
    args.save_name = best_model_dir
    logging.info(f"Created directory for saving models: {best_model_dir}")

    # --------------------------------------------------
    # DataLoader + Encoders (on-the-fly)
    # --------------------------------------------------
    train_loader, valid_loader, test_loader, encoders = encode_pretrained_feature(args)
    prot_encoder, drug_encoder = encoders

    # --------------------------------------------------
    # Build model (EncoderWrapper + ExplainPLI)
    # --------------------------------------------------
    pli_model = ExplainPLI(
        prot_out_dim=1280,
        drug_out_dim=768,
        args=args
    ).to(device)

    model = EncoderWrapper(
        prot_encoder=prot_encoder,
        drug_encoder=drug_encoder,
        pli_model=pli_model,
    ).to(device)

    # --------------------------------------------------
    # Load or train
    # --------------------------------------------------
    if os.path.exists(best_model_path):
        logging.info("Best model found. Loading the model...")
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        logging.info("Model loaded successfully.")
    else:
        logging.info("No saved model found. Proceeding with training...")
    
        optimizer = optim.AdamW(
            [
                {"params": model.prot_encoder.parameters(), "lr": args.lr * 0.1},
                {"params": model.drug_encoder.parameters(), "lr": args.lr * 0.1},
                {"params": model.pli_model.parameters(),   "lr": args.lr},
            ],
            weight_decay=1e-4,
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        criterion = nn.BCELoss()
    
        logging.info("Starting training...")
        best_state_dict = train(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            scheduler,
            best_model_path,
            device,
            args
        )
    
        logging.info("Training completed. Loading best model from disk...")
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        logging.info("Best model loaded successfully.")


    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    logging.info("Testing the model...")
    test(model, test_loader, device, args)

    wandb.finish()

if __name__ == "__main__":
    main()
