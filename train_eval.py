# train_eval.py
import os
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from KANBind import MultiBranchFusionModel


# -------------------------
# Early Stopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
                )
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# -------------------------
# Data loading
# -------------------------
def load_multiline_features(file_path, expected_dim):
    features, current_vector = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_vector and len(current_vector) == expected_dim:
                    features.append(np.array(current_vector))
                current_vector = []
            else:
                try:
                    current_vector.extend([float(val) for val in line.split()])
                except ValueError:
                    continue
    if current_vector and len(current_vector) == expected_dim:
        features.append(np.array(current_vector))
    return np.array(features)


def load_t5_features_from_pkl(file_path):
    with open(file_path, "rb") as tf:
        feature_dict = pickle.load(tf)
    return np.array([item for item in feature_dict.values()])


def load_data_for_multibranch(t5_p_path, t5_n_path, pssm_p_path, pssm_n_path, nmbac_p_path, nmbac_n_path):
    t5_pos = load_t5_features_from_pkl(t5_p_path)
    t5_neg = load_t5_features_from_pkl(t5_n_path)

    nmbac_pos = load_multiline_features(nmbac_p_path, 200)
    nmbac_neg = load_multiline_features(nmbac_n_path, 200)

    pssm_pos = pd.read_csv(pssm_p_path, header=None, skiprows=1).values
    pssm_neg = pd.read_csv(pssm_n_path, header=None, skiprows=1).values

    assert len(t5_pos) == len(pssm_pos) == len(nmbac_pos)
    assert len(t5_neg) == len(pssm_neg) == len(nmbac_neg)

    all_t5 = np.vstack([t5_pos, t5_neg])
    all_pssm = np.vstack([pssm_pos, pssm_neg])
    all_nmbac = np.vstack([nmbac_pos, nmbac_neg])

    labels = np.array([1] * len(t5_pos) + [0] * len(t5_neg))

    print(
        f"Data from {os.path.basename(t5_p_path).split('_')[0]} loaded: "
        f"{len(t5_pos)} Pos, {len(t5_neg)} Neg. Total: {len(labels)}"
    )
    return all_t5, all_pssm, all_nmbac, labels


# -------------------------
# Dataset / Dataloader
# -------------------------
class ProteinDataset(Dataset):
    def __init__(self, t5, pssm, nmbac, labels):
        self.t5 = t5
        self.pssm = pssm
        self.nmbac = nmbac
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.t5[idx], dtype=torch.float32),
            torch.tensor(self.pssm[idx], dtype=torch.float32),
            torch.tensor(self.nmbac[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def collate_fn_simple(batch):
    t5s, pssms, nmbacs, labels = zip(*batch)
    return torch.stack(t5s), torch.stack(pssms), torch.stack(nmbacs), torch.stack(labels)


# -------------------------
# Prevalence-calibrated utility metrics (paper)
# -------------------------
def prevalence_adjusted_precision(tpr: float, fpr: float, phi: float) -> float:
    """
    P_phi = (phi * TPR) / (phi * TPR + (1 - phi) * FPR)
    """
    denom = (phi * tpr) + ((1.0 - phi) * fpr)
    return (phi * tpr) / denom if denom > 0 else 0.0


# -------------------------
# Train / Eval
# -------------------------
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, reg_coef):
    model.train()
    total_loss = 0.0

    for t5, pssm, nmbac, labels in dataloader:
        t5 = t5.to(device)
        pssm = pssm.to(device)
        nmbac = nmbac.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(t5, pssm, nmbac).squeeze()
            base_loss = criterion(outputs, labels.float())
            reg_loss = model.classifier.regularization_loss()
            loss = base_loss + reg_coef * reg_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


def evaluate_for_paper_metrics(
    model,
    dataloader,
    criterion,
    device,
    threshold: float = 0.5,
    phis=(0.10, 0.03),
):
    """
    Returns metrics aligned with paper table:
    SN & SP & P_0.10 & FDR_0.10 & P_0.03 & FDR_0.03
    """
    model.eval()
    total_val_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for t5, pssm, nmbac, labels in dataloader:
            t5 = t5.to(device)
            pssm = pssm.to(device)
            nmbac = nmbac.to(device)
            labels = labels.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(t5, pssm, nmbac).squeeze()
                loss = criterion(logits, labels.float())

            total_val_loss += float(loss.item())

            probs = torch.sigmoid(logits)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_probs_np = np.asarray(all_probs, dtype=float)
    all_labels_np = np.asarray(all_labels, dtype=int)
    pred_bin = (all_probs_np >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels_np, pred_bin, labels=[0, 1]).ravel()

    # SN, SP
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    out = {
        "loss": total_val_loss / max(1, len(dataloader)),
        "SN": sn,
        "SP": sp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

    for phi in phis:
        p_phi = prevalence_adjusted_precision(tpr=sn, fpr=fpr, phi=float(phi))
        fdr_phi = 1.0 - p_phi
        # Use exact keys matching your paper notation
        out[f"P_{phi:.2f}"] = p_phi
        out[f"FDR_{phi:.2f}"] = fdr_phi

    return out


# -------------------------
# Main
# -------------------------
def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 256
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.4
    N_SPLITS = 5
    MAX_EPOCHS_CV = 200
    PATIENCE = 20
    RANDOM_STATE = 42

    # Paper metrics settings
    THRESHOLD = 0.5
    PHIS = (0.10, 0.03)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_files = {
     
    }
    test_files = {

    }

    print("\nLoading datasets...")
    train_t5_orig, train_pssm_orig, train_nmbac_orig, train_labels = load_data_for_multibranch(**train_files)
    test_t5_orig, test_pssm_orig, test_nmbac_orig, test_labels = load_data_for_multibranch(**test_files)

    print("\nStandardizing features...")
    scaler_t5 = StandardScaler().fit(train_t5_orig)
    scaler_pssm = StandardScaler().fit(train_pssm_orig)
    scaler_nmbac = StandardScaler().fit(train_nmbac_orig)

    train_t5 = scaler_t5.transform(train_t5_orig)
    test_t5 = scaler_t5.transform(test_t5_orig)
    train_pssm = scaler_pssm.transform(train_pssm_orig)
    test_pssm = scaler_pssm.transform(test_pssm_orig)
    train_nmbac = scaler_nmbac.transform(train_nmbac_orig)
    test_nmbac = scaler_nmbac.transform(test_nmbac_orig)

    train_dataset = ProteinDataset(train_t5, train_pssm, train_nmbac, train_labels)
    test_dataset = ProteinDataset(test_t5, test_pssm, test_nmbac, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_simple)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # class imbalance handling
    num_pos = int(np.sum(train_labels))
    num_neg = int(len(train_labels) - num_pos)
    pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE) if num_pos > 0 else torch.tensor([1.0], device=DEVICE)
    print(f"Class weights calculated. Positive class weight: {pos_weight.item():.4f}")

    print("\n" + "#" * 70 + "\n### PART 1: 5-Fold Cross-Validation ###\n" + "#" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    best_epochs_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
        print("\n" + "=" * 25 + f" Fold {fold + 1}/{N_SPLITS} " + "=" * 25)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_simple)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_simple)

        model = MultiBranchFusionModel(hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=f"cv_checkpoint_fold_{fold + 1}.pt")

        best_epoch_for_fold = 0
        for epoch in range(1, MAX_EPOCHS_CV + 1):
            reg_coef = max(0.001, 0.01 * (1 - epoch / MAX_EPOCHS_CV))
            _ = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, reg_coef)

            val_metrics = evaluate_for_paper_metrics(
                model, val_loader, criterion, DEVICE, threshold=THRESHOLD, phis=PHIS
            )
            val_loss = val_metrics["loss"]
            scheduler.step(val_loss)

            if val_loss < early_stopping.val_loss_min:
                best_epoch_for_fold = epoch

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch_for_fold}.")
                break

        best_epochs_list.append(best_epoch_for_fold)

        # load best model of this fold and evaluate again
        model.load_state_dict(torch.load(f"cv_checkpoint_fold_{fold + 1}.pt", map_location=DEVICE))
        val_metrics = evaluate_for_paper_metrics(
            model, val_loader, criterion, DEVICE, threshold=THRESHOLD, phis=PHIS
        )

        # Console print (exact paper metric set)
        print(
            f"Fold {fold + 1} Val | "
            f"SN={val_metrics['SN']:.4f} SP={val_metrics['SP']:.4f} "
            f"P_0.10={val_metrics['P_0.10']:.4f} FDR_0.10={val_metrics['FDR_0.10']:.4f} "
            f"P_0.03={val_metrics['P_0.03']:.4f} FDR_0.03={val_metrics['FDR_0.03']:.4f}"
        )

        fold_rows.append(
            [
                val_metrics["SN"],
                val_metrics["SP"],
                val_metrics["P_0.10"],
                val_metrics["FDR_0.10"],
                val_metrics["P_0.03"],
                val_metrics["FDR_0.03"],
            ]
        )

    results_df = pd.DataFrame(
        fold_rows,
        columns=["SN", "SP", "P_0.10", "FDR_0.10", "P_0.03", "FDR_0.03"],
    )

    final_train_epochs = int(np.ceil(np.mean(best_epochs_list))) if best_epochs_list else 25

    print("\n--- CV Summary (Paper Metrics) ---")
    print(results_df.agg(["mean", "std"]))

    print("\n" + "#" * 70 + f"\n### PART 2: Final Training for {final_train_epochs} Epochs ###\n" + "#" * 70)

    final_model = MultiBranchFusionModel(hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_simple)

    for epoch in range(1, final_train_epochs + 1):
        reg_coef = max(0.001, 0.01 * (1 - epoch / final_train_epochs))
        train_loss = train_epoch(final_model, full_train_loader, criterion, optimizer, scaler, DEVICE, reg_coef)
        print(f"Final Model Training - Epoch {epoch}/{final_train_epochs} | Train Loss: {train_loss:.4f}")

    print("\n" + "#" * 70 + "\n### PART 3: Final Evaluation on Independent Test Set ###\n" + "#" * 70)

    final_eval_criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate_for_paper_metrics(
        final_model, test_loader, final_eval_criterion, DEVICE, threshold=THRESHOLD, phis=PHIS
    )

    # Final print in exact paper order/fields
    print("\n" + "=" * 70)
    print("Final Performance (Independent Test) — Paper Metrics")
    print("=" * 70)
    print(f"SN        : {test_metrics['SN']:.4f}")
    print(f"SP        : {test_metrics['SP']:.4f}")
    print(f"P_0.10    : {test_metrics['P_0.10']:.4f}")
    print(f"FDR_0.10  : {test_metrics['FDR_0.10']:.4f}")
    print(f"P_0.03    : {test_metrics['P_0.03']:.4f}")
    print(f"FDR_0.03  : {test_metrics['FDR_0.03']:.4f}")
    print(f"ConfMat   : TP={test_metrics['TP']} FP={test_metrics['FP']} TN={test_metrics['TN']} FN={test_metrics['FN']}")
    print("=" * 70)

    # Optional: save CV summary to CSV (handy for paper table)
    results_df.to_csv("cv_paper_metrics_summary.csv", index=False)
    print("\nSaved: cv_paper_metrics_summary.csv")


if __name__ == "__main__":
    main()
