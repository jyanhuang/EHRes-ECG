import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import MITBIHAAMIDataset
from model import EHRes
from utils import compute_metrics, seed_everything


# ===============================
# configurations
# ===============================

DATA_ROOT = r".\datasets\mitdb"

MODEL_PATH = "./checkpointsV2/best_model.pt"

SAVE_DIR = "./test_results"

BATCH_SIZE = 128
NUM_WORKERS = 4
BEAT_LEN = 360
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# ===============================
# test function
# ===============================

@torch.no_grad()
def test(model, loader, criterion, device, save_dir):

    model.eval()

    total_loss = 0
    y_true = []
    y_pred = []

    for x, y in tqdm(loader, desc="Testing"):

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("True distribution:", Counter(y_true))
    print("Pred distribution:", Counter(y_pred))

    labels = ["N", "S", "V", "F", "Q"]

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    cm_df.to_csv(save_dir / "confusion_matrix.csv")

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), cm_sum, where=(cm_sum != 0))

    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv")

    print("\nConfusion matrix saved.")

    return metrics


# ===============================
# main function
# ===============================

def main():

    seed_everything(SEED)

    device = torch.device(DEVICE)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EHRes Model Testing")
    print("="*80)

    print("Model:", MODEL_PATH)
    print("Device:", device)

    # dataset
    test_dataset = MITBIHAAMIDataset(
        root_dir=DATA_ROOT,
        split="test",
        beat_len=BEAT_LEN,
        cache=True,
    )

    print("Test samples:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # model
    model = EHRes(num_classes=5).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded successfully.")

    criterion = nn.CrossEntropyLoss()

    metrics = test(model, test_loader, criterion, device, save_dir)

    print("\nFinal Test Metrics")
    print("-"*40)
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
