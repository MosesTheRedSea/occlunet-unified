import os
import json
import torch
import re
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.occulnet import OcculNetV2
from torchmetrics import ConfusionMatrix
from collections import Counter

"""
Step 1
Split recordings into train/val

Step 2
Apply augmentation ONLY to train recordings

Step 3
Never augment validation recordings
"""

class OcculDataset(Dataset):
    def __init__(self, folders, det_mapping, mat_mapping, obj_to_mat):

        self.folders     = folders
        self.det_mapping = det_mapping
        self.mat_mapping = mat_mapping
        self.obj_to_mat  = obj_to_mat

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        with open(folder / "metadata.json", "r") as f:
            meta = json.load(f)

        ir_list = []
        spec_list = []
        for ch in range(1, 17):
            ir_list.append(np.load(folder / f"ir_mic_{ch}.npy"))
            spec_list.append(np.load(folder / f"spec_mic_{ch}.npy"))

        ir_tensor = torch.from_numpy(np.stack(ir_list)).float()
        spec_tensor = torch.from_numpy(np.stack(spec_list)).float()

        t_det = torch.tensor(self.det_mapping[meta["object_type"]]).long()

        t_dist = torch.tensor(meta["occlusion_distance"]).float()

        mat_key = self.obj_to_mat[meta["object_type"]]
        t_mat   = torch.tensor(self.mat_mapping[mat_key]).long()

        return ir_tensor, spec_tensor, t_det, t_dist, t_mat
    
class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al. 2018).
    Adds an optional orthogonality term from the model.
    """
 
    def __init__(self, ortho_lambda=0.01):
        super().__init__()
        self.log_vars     = nn.Parameter(torch.zeros(3))
        self.ortho_lambda = ortho_lambda
 
    def forward(self, p_det, t_det, p_dist, t_dist, p_mat, t_mat,
                ortho_loss=None):
 
        loss_det  = nn.functional.nll_loss(p_det, t_det)
        loss_mat  = nn.functional.nll_loss(p_mat, t_mat)
        loss_dist = nn.functional.mse_loss(p_dist.squeeze(), t_dist)
 
        prec_det  = torch.exp(-self.log_vars[0])
        prec_dist = torch.exp(-self.log_vars[1])
        prec_mat  = torch.exp(-self.log_vars[2])
 
        total = (prec_det  * loss_det  + self.log_vars[0] +
                 prec_dist * loss_dist + self.log_vars[1] +
                 prec_mat  * loss_mat  + self.log_vars[2])
 
        if ortho_loss is not None:
            total = total + self.ortho_lambda * ortho_loss
 
        return total, (loss_det.item(), loss_dist.item(), loss_mat.item())

def run_evaluation(model, loader, device, det_classes, mat_classes, save_dir):
    model.eval()
    all_det_p, all_det_t = [], []
    all_mat_p, all_mat_t = [], []
    all_dist_p, all_dist_t = [], []

    with torch.no_grad():
        for ir, spec, t_det, t_dist, t_mat in loader:
            ir, spec = ir.to(device), spec.to(device)
            p_det, p_dist, p_mat = model(ir, spec)
            all_det_p.extend(torch.argmax(p_det, dim=1).cpu().numpy())
            all_det_t.extend(t_det.numpy())
            all_mat_p.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
            all_mat_t.extend(t_mat.numpy())
            all_dist_p.extend(p_dist.squeeze().cpu().numpy())
            all_dist_t.extend(t_dist.numpy())

    acc_det  = np.mean(np.array(all_det_p) == np.array(all_det_t)) * 100
    rmse_dist = np.sqrt(mean_squared_error(all_dist_t, all_dist_p))
    mae_dist  = mean_absolute_error(all_dist_t, all_dist_p)

    print(f"\nDetection Accuracy : {acc_det:.2f}%")
    print(f"Distance RMSE      : {rmse_dist:.2f} m")
    print(f"Distance MAE       : {mae_dist:.2f} m")

    # --- Detection confusion matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_det = confusion_matrix(all_det_t, all_det_p, normalize='true')
    sns.heatmap(cm_det, annot=True, fmt='.2f', ax=ax,
                xticklabels=det_classes, yticklabels=det_classes)
    ax.set_title("Detection Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_dir / "detection_cm.png", dpi=150)
    plt.close()

    # --- Material confusion matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_mat = confusion_matrix(all_mat_t, all_mat_p, normalize='true')
    sns.heatmap(cm_mat, annot=True, fmt='.2f', ax=ax,
                xticklabels=mat_classes, yticklabels=mat_classes)
    ax.set_title("Material Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_dir / "material_cm.png", dpi=150)
    plt.close()

    # --- Distance scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_dist_t, all_dist_p, alpha=0.5, color='teal')
    ax.plot([min(all_dist_t), max(all_dist_t)],
            [min(all_dist_t), max(all_dist_t)], 'r--', label='Perfect prediction')
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Predicted distance (m)")
    ax.set_title(f"Distance  RMSE={rmse_dist:.3f}m  MAE={mae_dist:.3f}m")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "distance_scatter.png", dpi=150)
    plt.close()

    # --- Metrics text ---
    with open(save_dir / "results.txt", "w") as f:
        f.write(f"Detection Accuracy : {acc_det:.2f}%\n")
        f.write(f"Distance RMSE      : {rmse_dist:.2f} m\n")
        f.write(f"Distance MAE       : {mae_dist:.2f} m\n")

def epoch_metrics(model, loader, device):
    model.eval()

    det_preds, det_targets = [], []
    mat_preds, mat_targets = [], []
    dist_preds, dist_targets = [], []

    with torch.no_grad():
        for ir, spec, t_det, t_dist, t_mat in loader:
            ir = ir.to(device)
            spec = spec.to(device)

            p_det, p_dist, p_mat = model(ir, spec)

            det_preds.extend(torch.argmax(p_det, dim=1).cpu().numpy())
            det_targets.extend(t_det.numpy())

            mat_preds.extend(torch.argmax(p_mat, dim=1).cpu().numpy())
            mat_targets.extend(t_mat.numpy())

            dist_preds.extend(p_dist.squeeze().cpu().numpy())
            dist_targets.extend(t_dist.numpy())

    det_acc = np.mean(np.array(det_preds) == np.array(det_targets)) * 100
    mat_acc = np.mean(np.array(mat_preds) == np.array(mat_targets)) * 100
    rmse = np.sqrt(mean_squared_error(dist_targets, dist_preds))
    mae = mean_absolute_error(dist_targets, dist_preds)

    return det_acc, mat_acc, rmse, mae

def get_next_run_folder(base_path):
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)

    runs = []

    for folder in base_path.iterdir():
        if folder.is_dir():
            match = re.match(r"run_(\d+)", folder.name)
            if match:
                runs.append(int(match.group(1)))

    next_run = max(runs, default=0) + 1

    run_path = base_path / f"run_{next_run}"
    run_path.mkdir()

    return run_path

if __name__ == '__main__':

    # processed_root = Path("./data/processed")

    # occlusion_classes = Counter()
    # object_classes = Counter()

    # for folder in processed_root.iterdir():
    #     if folder.is_dir() and (folder / "metadata.json").exists():
    #         with open(folder / "metadata.json", "r") as f:
    #             meta = json.load(f)

    #         occlusion_classes[meta["occlusion_type"]] += 1
    #         object_classes[meta["object_type"]] += 1


    # print("\nOcclusion classes:")
    # for k, v in occlusion_classes.items():
    #     print(f"{k}: {v}")

    # print("\nObject/material classes:")
    # for k, v in object_classes.items():
    #     print(f"{k}: {v}")

    OBJECT_TO_MATERIAL = {
        "no_object": "none",
        "cardboard":  "cardboard",
        "sandbag":    "sand",
        "speaker":    "synthetic",
        "plastic":    "plastic",
    }

    obj_classes = ["no_object", "cardboard", "sandbag", "speaker", "plastic"] 
    mat_classes = ["none", "cardboard", "sand", "plastic", "synthetic"]

    det_map = {name: i for i, name in enumerate(obj_classes)}
    mat_map = {name: i for i, name in enumerate(mat_classes)}

    print(mat_classes)
    print(obj_classes)
    print(det_map)
    print(mat_map)

    processed_root = Path("./data/processed/")
    augmented_root = Path("./data/augmented/")

    check = set()

    # Original recordings only
    original_folders = sorted([
        f for f in processed_root.iterdir()
        if f.is_dir() and (f / "metadata.json").exists()
    ])

    

    # Temporary dataset for labels
    temp_dataset = OcculDataset(
        original_folders,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    labels = [
        temp_dataset[i][2].item()
        for i in range(len(temp_dataset))
    ]

    # Split ORIGINAL recordings first
    train_folders, val_folders = train_test_split(
        original_folders,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    # Add augmentations ONLY to training set
    train_full = []
    for folder in train_folders:
        train_full.append(folder)
        name = folder.name
        for aug in ["noise", "shift", "scale"]:
            aug_folder = augmented_root / f"{aug}_{name}"
            if aug_folder.exists():
                train_full.append(aug_folder)

    # Validation remains ORIGINAL ONLY
    val_full = val_folders

    train_dataset = OcculDataset(
        train_full,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    val_dataset = OcculDataset(
        val_full,
        det_mapping=det_map,
        mat_mapping=mat_map,
        obj_to_mat=OBJECT_TO_MATERIAL
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    print(f"Original recordings: {len(original_folders)}")
    print(f"Training originals: {len(train_folders)}")
    print(f"Validation originals: {len(val_folders)}")
    print(f"Training samples after augmentation: {len(train_full)}")

    aug_count = len(train_full) - len(train_folders)
    print(f"Augmented samples added: {aug_count}")

    # dataset = Dataset(root_dir="./processed", det_mapping=det_map, mat_mapping=mat_map)
    # processed_root = Path("./processed")

    # original_folders = [
    #     f for f in processed_root.iterdir()
    #     if f.is_dir()
    # ]

    # indices = np.arange(len(dataset))
    # labels = [dataset[i][2].item() for i in range(len(dataset))]

    # train_folders, val_folders = train_test_split(
    #     original_folders,
    #     test_size=0.15,
    #     stratify=labels,
    #     random_state=42
    # )

    # train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    # val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 15
    ORTHO_LAMBDA = 0.01

    MODEL_DIR = (
        "./models"
    )

    RESULT_DIR = (
        "./results"
    )

    model_run_dir = get_next_run_folder(MODEL_DIR)
    result_run_dir = get_next_run_folder(RESULT_DIR)

    print(f"Saving model to: {model_run_dir}")
    print(f"Saving results to: {result_run_dir}")


    model = OcculNetV2(len(obj_classes), len(mat_classes)).to(DEVICE)
    criterion = MultiTaskLoss(ortho_lambda=ORTHO_LAMBDA).to(DEVICE)
 

    optimizer = optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': criterion.parameters(), 'lr': LR},
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_score = -float("inf")
    early_stop_cnt  = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        train_det_losses = []
        train_dist_losses = []
        train_mat_losses = []
        
        for ir, spec, t_det, t_dist, t_mat in train_loader:

            ir, spec = ir.to(DEVICE), spec.to(DEVICE)

            t_det, t_dist, t_mat = t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)

            optimizer.zero_grad()
            p_det, p_dist, p_mat = model(ir, spec)

            loss, (det_loss, dist_loss, mat_loss) = criterion(
                p_det, t_det,
                p_dist, t_dist,
                p_mat, t_mat,
                ortho_loss=model.orthogonality_loss
            )

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            train_det_losses.append(det_loss)
            train_dist_losses.append(dist_loss)
            train_mat_losses.append(mat_loss)

        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for ir, spec, t_det, t_dist, t_mat in val_loader:
                ir, spec = ir.to(DEVICE), spec.to(DEVICE)
                t_det, t_dist, t_mat = (
                    t_det.to(DEVICE), t_dist.to(DEVICE), t_mat.to(DEVICE)
                )
                p_det, p_dist, p_mat = model(ir, spec)
                v_loss, _ = criterion(
                    p_det,
                    t_det,
                    p_dist,
                    t_dist,
                    p_mat,
                    t_mat,
                    ortho_loss=model.orthogonality_loss
                )
                val_losses.append(v_loss.item())
 
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_train_det = np.mean(train_det_losses)
        avg_train_dist = np.mean(train_dist_losses)
        avg_train_mat = np.mean(train_mat_losses)

        det_acc, mat_acc, rmse, mae = epoch_metrics(model, val_loader, DEVICE)

        sigmas = (
            torch.exp(0.5 * criterion.log_vars)
            .detach()
            .cpu()
            .numpy()
        )

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Det: {det_acc:.1f}% | Mat: {mat_acc:.1f}% | "
            f"RMSE: {rmse:.3f}m | MAE: {mae:.3f}m | "
            f"DetLoss: {avg_train_det:.3f} | "
            f"DistLoss: {avg_train_dist:.3f} | "
            f"MatLoss: {avg_train_mat:.3f} | "
            f"σ=[{sigmas[0]:.2f}, {sigmas[1]:.2f}, {sigmas[2]:.2f}] | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        scheduler.step()
 
        score = (
            0.4 * (det_acc / 100.0) +
            0.4 * (mat_acc / 100.0) -
            0.2 * rmse
        )

        if score > best_score:
            best_score = score
            early_stop_cnt = 0

            torch.save(
                model.state_dict(),
                model_run_dir / "best_model.pth"
            )
 
        else:
            early_stop_cnt += 1

            if early_stop_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
 
    # Always save the final checkpoint too
    torch.save(model.state_dict(), model_run_dir / "final_model.pth")
 
    # Load best weights for evaluation
    model.load_state_dict(torch.load(model_run_dir / "best_model.pth"))
    run_evaluation(model, val_loader, DEVICE, obj_classes, mat_classes, result_run_dir)
 