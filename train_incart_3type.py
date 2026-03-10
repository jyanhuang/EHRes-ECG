import argparse
from email.policy import default
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset_incart_3type import (
    INCARTDataset2Cls,
    INCARTDataset3Cls,
    INCARTDataset5Cls,
    list_incart_records,
    INCARTDatasetHF3Cls,
)
from model import EHRes
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

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


def parse_args():
    parser = argparse.ArgumentParser()

    # basic settings
    parser.add_argument("--data_root", type=str, default=r"./datasets/incart", help="INCART root directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_incart/hf3cls")
    # parser.add_argument("--task", type=str, default="5cls", choices=["2cls", "3cls", "5cls"])
    parser.add_argument(
        "--task",
        type=str,
        default="2cls",
        choices=["2cls", "3cls", "5cls", "hf3cls"]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beat_len", type=int, default=360)
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)

    # 5-fold settings
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--fold", type=int, default=1, help="Run only one fold: 1~n_splits. Default -1 means run all folds.")

    # imbalance handling
    parser.add_argument("--use_class_weight", action="store_true", help="Use class-weighted cross entropy")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use WeightedRandomSampler for train set")

    # pruning settings
    parser.add_argument("--enable_pruning", action="store_true")
    parser.add_argument("--prune_epoch", type=int, default=12)
    parser.add_argument("--network_prune_amount", type=float, default=0.9)
    parser.add_argument("--block_prune_amount", type=float, default=0.9)

    return parser.parse_args()


def get_task_config(task: str):
    if task == "2cls":
        dataset_cls = INCARTDataset2Cls
        label_names = ["Normal", "Abnormal"]
        num_classes = 2

    elif task == "3cls":
        dataset_cls = INCARTDataset3Cls
        label_names = ["N", "S", "V"]
        num_classes = 3

    elif task == "5cls":
        dataset_cls = INCARTDataset5Cls
        label_names = ["N", "V", "R", "A", "F"]
        num_classes = 5

    elif task == "hf3cls":
        dataset_cls = INCARTDatasetHF3Cls
        label_names = ["N", "V", "A"]
        num_classes = 3

    else:
        raise ValueError(f"Unsupported task: {task}")

    label2idx = {name: i for i, name in enumerate(label_names)}
    idx2label = {i: name for i, name in enumerate(label_names)}

    return {
        "dataset_cls": dataset_cls,
        "label_names": label_names,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "num_classes": num_classes,
    }


def build_class_weights_from_dataset(dataset, device):
    counts_dict = dataset.get_class_counts()
    label_names = list(dataset.LABEL2IDX.keys())
    class_counts = np.array([counts_dict[name] for name in label_names], dtype=np.float32)

    if np.any(class_counts == 0):
        raise ValueError(f"Found empty class in training set: {class_counts.tolist()}")

    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    return class_counts, class_weights


def build_weighted_sampler_from_dataset(dataset):
    train_labels = dataset.y
    num_classes = len(dataset.LABEL2IDX)

    class_sample_count = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    if np.any(class_sample_count == 0):
        raise ValueError(f"Found empty class in training set: {class_sample_count.tolist()}")

    sample_weights = 1.0 / class_sample_count[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


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
def evaluate(model, loader, criterion, device, save_dir, label_names):
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

    label_ids = list(range(len(label_names)))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    print("True distribution:", Counter(y_true))
    print("Pred distribution:", Counter(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    # cm_df.to_csv(save_dir / "confusion_matrix.csv", index=True)
    cm_df.to_csv(
        save_dir / "confusion_matrix.csv",
        mode='a',
        header=False,
        index=True
    )

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), cm_sum, where=(cm_sum != 0))
    cm_norm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
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


def build_dataloaders(args, dataset_cls, train_records, val_records):
    train_dataset = dataset_cls(
        root_dir=args.data_root,
        records=train_records,
        beat_len=args.beat_len,
        lead=args.lead,
        normalize=True,
        cache=True,
    )

    val_dataset = dataset_cls(
        root_dir=args.data_root,
        records=val_records,
        beat_len=args.beat_len,
        lead=args.lead,
        normalize=True,
        cache=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print("Train class counts:", train_dataset.get_class_counts())
    print("Val class counts  :", val_dataset.get_class_counts())

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler_from_dataset(train_dataset)
        train_loader = DataLoader(
            **train_loader_kwargs,
            sampler=sampler,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            **train_loader_kwargs,
            shuffle=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def run_one_fold(args, fold_id, train_records, val_records, device, root_save_dir, task_cfg):
    fold_save_dir = root_save_dir / f"fold_{fold_id}"
    fold_save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Fold {fold_id}")
    print("=" * 80)
    print("Train records:", train_records)
    print("Val records  :", val_records)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        args=args,
        dataset_cls=task_cfg["dataset_cls"],
        train_records=train_records,
        val_records=val_records,
    )

    model = EHRes(
        num_classes=task_cfg["num_classes"],
        dropout_p=args.dropout_p,
    ).to(device)
    print(f"Trainable params before pruning: {count_trainable_params(model):,}")

    if args.use_class_weight:
        class_counts, class_weights = build_class_weights_from_dataset(train_dataset, device)
        print("Class counts  :", class_counts.tolist())
        print("Class weights :", class_weights.detach().cpu().numpy().round(4).tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
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

    best_f1 = -1.0
    pruning_applied = False
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta
    )

    best_metrics = None

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
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            save_dir=fold_save_dir,
            label_names=task_cfg["label_names"],
        )

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
            best_metrics = val_metrics

            best_ckpt = {
                "epoch": epoch,
                "fold": fold_id,
                "task": args.task,
                "best_val_f1": best_f1,
                "model_state_dict": export_state_dict(model),
                "args": vars(args),
                "label_map": task_cfg["idx2label"],
                "train_records": train_records,
                "val_records": val_records,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            best_model_path = fold_save_dir / "best_model.pt"
            save_checkpoint(best_ckpt, str(best_model_path))
            print(f"[INFO] Best model saved to: {best_model_path}")

        if epoch % 10 == 0:
            epoch_ckpt = {
                "epoch": epoch,
                "fold": fold_id,
                "task": args.task,
                "best_val_f1": best_f1,
                "model_state_dict": export_state_dict(model),
                "args": vars(args),
                "label_map": task_cfg["idx2label"],
                "train_records": train_records,
                "val_records": val_records,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            epoch_model_path = fold_save_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(epoch_ckpt, str(epoch_model_path))
            print(f"[INFO] Epoch checkpoint saved to: {epoch_model_path}")

        if early_stopper.step(val_metrics["f1"]):
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

        print("-" * 80)

    print(f"Fold {fold_id} finished. Best val F1 = {best_f1:.4f}")
    return {
        "fold": fold_id,
        "best_f1": best_f1,
        "best_metrics": best_metrics,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    task_cfg = get_task_config(args.task)

    device = torch.device(args.device)
    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EHRes Training on INCART")
    print("=" * 80)
    print(f"Task       : {args.task}")
    print(f"Labels     : {task_cfg['label_names']}")
    print(f"Data root  : {args.data_root}")
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Beat len   : {args.beat_len}")
    print(f"Lead       : {args.lead}")
    print(f"Dropout    : {args.dropout_p}")
    print(f"5-Fold     : {args.n_splits}")
    print(f"Shuffle    : {args.shuffle}")
    print(f"Cls Weight : {args.use_class_weight}")
    print(f"Sampler    : {args.use_weighted_sampler}")
    print(f"Pruning    : {'Enabled' if args.enable_pruning else 'Disabled'}")
    if args.enable_pruning:
        print(f"Prune epoch          : {args.prune_epoch}")
        print(f"Network prune amount : {args.network_prune_amount}")
        print(f"Block prune amount   : {args.block_prune_amount}")
    print("=" * 80)

    records = list_incart_records(Path(args.data_root))
    print(f"Total records: {len(records)}")
    print("All records:", records)

    kf = KFold(
        n_splits=args.n_splits,
        shuffle=args.shuffle,
        random_state=args.seed if args.shuffle else None,
    )

    all_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(records), start=1):
        if args.fold != -1 and fold_idx != args.fold:
            continue

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        fold_result = run_one_fold(
            args=args,
            fold_id=fold_idx,
            train_records=train_records,
            val_records=val_records,
            device=device,
            root_save_dir=save_dir,
            task_cfg=task_cfg,
        )
        all_results.append(fold_result)

    if len(all_results) == 0:
        print("No fold was run. Please check --fold setting.")
        return

    summary_rows = []
    for r in all_results:
        row = {
            "fold": r["fold"],
            "best_f1": r["best_f1"],
        }
        if r["best_metrics"] is not None:
            row.update({
                "accuracy": r["best_metrics"]["accuracy"],
                "precision": r["best_metrics"]["precision"],
                "recall": r["best_metrics"]["recall"],
                "f1": r["best_metrics"]["f1"],
                "loss": r["best_metrics"]["loss"],
            })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir / "5fold_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("5-Fold Summary")
    print("=" * 80)
    print(summary_df)

    if "best_f1" in summary_df.columns:
        print(f"\nMean Best F1: {summary_df['best_f1'].mean():.4f}")
        print(f"Std  Best F1: {summary_df['best_f1'].std():.4f}")

    print(f"\nSummary saved to: {save_dir / '5fold_summary.csv'}")


if __name__ == "__main__":
    main()