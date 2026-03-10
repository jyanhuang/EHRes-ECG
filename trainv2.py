import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset import MITBIHAAMIDataset, IDX2LABEL
from model import EHRes
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from utils import (
    apply_mlpo,
    compute_metrics,
    count_trainable_params,
    export_state_dict,
    save_checkpoint,
    seed_everything,
)


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop


def parse_args():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--data_root", type=str, default=r".\datasets\mitdb", help="MIT-BIH root directory")
    parser.add_argument("--save_dir", type=str, default="./checkpointsV2-TEST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--beat_len", type=int, default=360)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)

    # pruning settings
    parser.add_argument("--enable_pruning", action="store_true")
    parser.add_argument("--prune_epoch", type=int, default=12)
    parser.add_argument("--network_prune_amount", type=float, default=0.9)
    parser.add_argument("--block_prune_amount", type=float, default=0.9)

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, save_dir):
    model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("True distribution:", Counter(y_true))
    print("Pred distribution:", Counter(y_pred))

    labels = ["N", "S", "V", "F", "Q"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    # cm_df.to_csv(save_dir / "confusion_matrix.csv", index=True)
    cm_df.to_csv(
        save_dir / "confusion_matrix.csv",
        mode='a',
        header=False,
        index=True
    )
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), cm_sum, where=(cm_sum != 0))
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    # cm_norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv", index=True)
    cm_norm_df.to_csv(
        save_dir / "confusion_matrix_normalized.csv",
        mode='a',
        header=False,
        index=True
    )

    print(f"\nConfusion matrix saved to: {save_dir / 'confusion_matrix.csv'}")
    print(f"Normalized confusion matrix saved to: {save_dir / 'confusion_matrix_normalized.csv'}")

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EHRes Training")
    print("=" * 80)
    print(f"Data root  : {args.data_root}")
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Beat len   : {args.beat_len}")
    print(f"Pruning    : {'Enabled' if args.enable_pruning else 'Disabled'}")
    if args.enable_pruning:
        print(f"Prune epoch          : {args.prune_epoch}")
        print(f"Network prune amount : {args.network_prune_amount}")
        print(f"Block prune amount   : {args.block_prune_amount}")
    print("=" * 80)

    # Load DS1 as training pool, then split into train / val
    full_train_dataset = MITBIHAAMIDataset(
        root_dir=args.data_root,
        split="train",
        beat_len=args.beat_len,
        cache=True,
    )

    labels = full_train_dataset.y
    print(f"Total DS1 samples: {len(full_train_dataset)}")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.val_ratio,
        random_state=args.seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    train_labels = full_train_dataset.y[train_idx]
    class_counts = np.bincount(train_labels, minlength=5).astype(np.int64)
    print("Class counts:", class_counts)

    # 稳定版：先不用 WeightedRandomSampler，也先不用 class_weights
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model / Loss / Optimizer
    model = EHRes(num_classes=5).to(device)
    print(f"Trainable params before pruning: {count_trainable_params(model):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # Train
    best_f1 = -1.0
    pruning_applied = False
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta
    )

    for epoch in range(1, args.epochs + 1):
        if args.enable_pruning and (not pruning_applied) and epoch == args.prune_epoch:
            print("\n[INFO] Applying MLPO pruning...")
            apply_mlpo(
                model,
                network_prune_amount=args.network_prune_amount,
                block_prune_amount=args.block_prune_amount,
            )
            pruning_applied = True
            print(f"[INFO] Trainable params after pruning wrapper: {count_trainable_params(model):,}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, save_dir)

        scheduler.step(val_metrics["f1"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train Acc {train_metrics['accuracy']:.4f} | "
            f"Train Pre {train_metrics['precision']:.4f} | "
            f"Train Rec {train_metrics['recall']:.4f} | "
            f"Train F1 {train_metrics['f1']:.4f}"
        )

        print(
            f"                    "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val Pre {val_metrics['precision']:.4f} | "
            f"Val Rec {val_metrics['recall']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            best_ckpt = {
                "epoch": epoch,
                "best_val_f1": best_f1,
                "model_state_dict": export_state_dict(model),
                "args": vars(args),
                "label_map": IDX2LABEL,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            best_model_path = save_dir / "best_model.pt"
            save_checkpoint(best_ckpt, str(best_model_path))
            print(f"[INFO] Best model saved to: {best_model_path}")

        if epoch % 10 == 0:
            epoch_ckpt = {
                "epoch": epoch,
                "best_val_f1": best_f1,
                "model_state_dict": export_state_dict(model),
                "args": vars(args),
                "label_map": IDX2LABEL,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            epoch_model_path = save_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(epoch_ckpt, str(epoch_model_path))
            print(f"[INFO] Epoch checkpoint saved to: {epoch_model_path}")

        if early_stopper.step(val_metrics["f1"]):
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

        print("-" * 80)

    print(f"Training finished. Best val F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()