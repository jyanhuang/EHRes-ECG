import copy
import os
import random
from typing import Dict, List
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import BasicBlockHeartNet
import pywt


import numpy as np
import pywt


def wavelet_denoise_ecg(
    signal,
    wavelet: str = "sym8",
    level: int = 5,
    threshold_mode: str = "soft",
    threshold_method: str = "universal",
    preserve_approx: bool = True,
):
    """
    ECG denoising with wavelet decomposition.

    Args:
        signal: 1D ECG signal
        wavelet: wavelet basis, e.g. "sym8"
        level: decomposition level
        threshold_mode: "soft" or "hard"
        threshold_method: "universal" or "bayes"
        preserve_approx: whether to preserve approximation coefficients

    Returns:
        denoised 1D signal with same length as input
    """
    x = np.asarray(signal, dtype=np.float32).reshape(-1)

    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(x), wave.dec_len)
    level = min(level, max_level)

    if level < 1:
        return x.copy()

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)

    cA = coeffs[0]
    details = coeffs[1:]

    # estimate noise from highest-frequency detail coefficients
    finest_detail = details[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    sigma = max(float(sigma), 1e-8)

    denoised_coeffs = []
    denoised_coeffs.append(cA if preserve_approx else np.zeros_like(cA))

    n = len(x)

    for d in details:
        if threshold_method == "universal":
            thr = sigma * np.sqrt(2.0 * np.log(n))
        elif threshold_method == "bayes":
            var_d = np.var(d)
            thr = (sigma ** 2) / np.sqrt(max(var_d - sigma ** 2, 1e-8))
        else:
            raise ValueError(f"Unsupported threshold_method: {threshold_method}")

        d_new = pywt.threshold(d, value=thr, mode=threshold_mode)
        denoised_coeffs.append(d_new)

    rec = pywt.waverec(denoised_coeffs, wavelet=wavelet)
    rec = rec[: len(x)].astype(np.float32)
    return rec


def preprocess_ecg_signal(
    signal,
    use_denoise: bool = False,
    wavelet: str = "sym8",
    level: int = 5,
    threshold_mode: str = "soft",
    threshold_method: str = "universal",
    use_zscore: bool = True,
):
    """
    ECG preprocessing for ablation experiments:
    1) optional wavelet denoising
    2) optional z-score normalization
    """
    x = np.asarray(signal, dtype=np.float32)

    if use_denoise:
        x = wavelet_denoise_ecg(
            x,
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_method=threshold_method,
            preserve_approx=True,
        )

    if use_zscore:
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

    return x.astype(np.float32)


def normalize_signal_zscore(signal, eps: float = 1e-8):
    x = np.asarray(signal, dtype=np.float32)
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def preprocess_ecg_signal(
    signal,
    use_denoise: bool = True,
    wavelet: str = "sym8",
    level: int = 5,
    threshold_mode: str = "soft",
    threshold_method: str = "universal",
    use_zscore: bool = True,
):
    """
    One-stop ECG preprocessing:
    1) wavelet denoising
    2) z-score normalization
    """
    x = np.asarray(signal, dtype=np.float32)

    if use_denoise:
        x = wavelet_denoise_ecg(
            x,
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_method=threshold_method,
            preserve_approx=True,
        )

    if use_zscore:
        x = normalize_signal_zscore(x)

    return x.astype(np.float32)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss



def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_confusion(y_true: List[int], y_pred: List[int]):
    return confusion_matrix(y_true, y_pred)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_mlpo(
    model: nn.Module,
    network_prune_amount: float = 0.9,
    block_prune_amount: float = 0.9,
):
    """
    Multi-Level Pruning Optimization:
      - network-level: conv1 + fc
      - block-level: first conv1 in each REB
    """
    # network-level pruning
    prune.l1_unstructured(model.conv1, name="weight", amount=network_prune_amount)
    prune.l1_unstructured(model.fc, name="weight", amount=network_prune_amount)

    # block-level pruning
    for m in model.modules():
        if isinstance(m, BasicBlockHeartNet):
            prune.l1_unstructured(m.conv1, name="weight", amount=block_prune_amount)


def make_pruning_permanent(model: nn.Module):
    """
    Remove pruning re-parameterization and keep sparse weights as normal tensors.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")


def export_state_dict(model: nn.Module):
    """
    Save a clean state_dict even if the current model is still wrapped by pruning.
    """
    model_copy = copy.deepcopy(model).cpu()
    make_pruning_permanent(model_copy)
    return model_copy.state_dict()


def save_checkpoint(state: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

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

def measure_latency(model, device, input_shape=(1, 1, 360), warmup=50, runs=200):
    """
    测平均单次推理延迟（ms）
    input_shape: (B, C, L)
    """
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_latency_ms = (end - start) * 1000.0 / runs
    throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_sec": throughput,
    }


def measure_flops_and_params(model, device, input_shape):
    try:
        from thop import profile
    except ImportError:
        print("[WARN] thop is not installed. Please run: pip install thop")
        return None

    try:
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)

        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # 常见约定：FLOPs ≈ 2 x MACs

        return {
            "macs": macs,
            "flops": flops,
            "params": params,
        }
    except Exception as e:
        print(f"[WARN] measure_flops_and_params failed: {e}")
        return None

def format_flops(num):
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num / 1e9:.3f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return f"{num:.0f}"


def format_params(num):
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num / 1e9:.3f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return f"{num:.0f}"

def count_effective_nonzero_params(model):
    total = 0
    nonzero = 0

    for module in model.modules():
        if hasattr(module, "weight"):
            w = module.weight.detach()
            total += w.numel()
            nonzero += torch.count_nonzero(w).item()

        if hasattr(module, "bias") and module.bias is not None:
            b = module.bias.detach()
            total += b.numel()
            nonzero += torch.count_nonzero(b).item()

    sparsity = 1.0 - nonzero / total if total > 0 else 0.0
    return total, nonzero, sparsity