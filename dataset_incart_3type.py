import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError as e:
    raise ImportError(
        "wfdb is required. Please install it with: pip install wfdb"
    ) from e


def list_incart_records(root_dir: Path) -> List[str]:
    """
    Find all INCART record names from .dat files.
    Example:
        I01.dat -> I01
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    records = sorted([p.stem for p in root_dir.glob("*.dat")])
    if len(records) == 0:
        raise RuntimeError(f"No .dat records found in: {root_dir}")
    return records


class BaseINCARTDataset(Dataset):
    """
    Base dataset for INCART beat classification.

    Each sample is a beat-centered segment extracted around an annotated R-peak.
    Subclasses only need to provide:
        - SYMBOL2LABEL
        - LABEL2IDX
        - DATASET_NAME

    Expected folder structure:
        root/
          I01.dat
          I01.hea
          I01.atr
          ...
    """

    SYMBOL2LABEL: Dict[str, str] = {}
    LABEL2IDX: Dict[str, int] = {}
    IDX2LABEL: Dict[int, str] = {}
    DATASET_NAME: str = "incart_base"

    def __init__(
        self,
        root_dir: str,
        records: Optional[List[str]] = None,
        beat_len: int = 360,
        lead: int = 0,
        normalize: bool = True,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.beat_len = beat_len
        self.half_len = beat_len // 2
        self.lead = lead
        self.normalize = normalize
        self.cache = cache

        if records is None:
            records = list_incart_records(self.root_dir)

        self.records = sorted(records)
        self.x, self.y = self._load_or_process()

    def _symbol_to_index(self, symbol: str) -> Optional[int]:
        mapped_label = self.SYMBOL2LABEL.get(symbol, None)
        if mapped_label is None:
            return None
        return self.LABEL2IDX[mapped_label]

    def _cache_path(self) -> Path:
        cache_dir = self.root_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        record_str = "_".join(self.records)
        record_hash = hashlib.md5(record_str.encode("utf-8")).hexdigest()[:12]

        filename = (
            f"{self.DATASET_NAME}_lead{self.lead}_len{self.beat_len}_{record_hash}.npz"
        )
        return cache_dir / filename

    def _load_or_process(self) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = self._cache_path()

        if self.cache and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return data["x"].astype(np.float32), data["y"].astype(np.int64)

        beats: List[np.ndarray] = []
        labels: List[int] = []

        for rec_name in self.records:
            rec_path = str(self.root_dir / rec_name)

            record = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")

            if record.p_signal is None:
                continue

            if self.lead >= record.p_signal.shape[1]:
                raise ValueError(
                    f"Requested lead={self.lead}, but record {rec_name} only has "
                    f"{record.p_signal.shape[1]} leads."
                )

            signal = record.p_signal[:, self.lead]

            for r_loc, symbol in zip(ann.sample, ann.symbol):
                label_idx = self._symbol_to_index(symbol)
                if label_idx is None:
                    # skip symbols not used by current task
                    # e.g. '+', '[', ']', 'p', 't', '/', 'Q', etc.
                    continue

                left = r_loc - self.half_len
                right = r_loc + self.half_len

                if left < 0 or right > len(signal):
                    continue

                beat = signal[left:right].astype(np.float32)

                if len(beat) != self.beat_len:
                    continue

                if self.normalize:
                    mean = beat.mean()
                    std = beat.std() + 1e-8
                    beat = (beat - mean) / std

                beats.append(beat)
                labels.append(label_idx)

        if len(beats) == 0:
            raise RuntimeError(
                f"No valid beats found in {self.root_dir}. "
                f"Please check your INCART files and annotation files."
            )

        x = np.stack(beats, axis=0).astype(np.float32)
        y = np.array(labels, dtype=np.int64)

        if self.cache:
            np.savez_compressed(cache_path, x=x, y=y)

        return x, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        beat = torch.from_numpy(self.x[idx]).unsqueeze(0)  # [1, L]
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return beat, label

    def get_class_counts(self) -> Dict[str, int]:
        counts = {k: 0 for k in self.LABEL2IDX.keys()}
        unique, cnt = np.unique(self.y, return_counts=True)
        for u, c in zip(unique.tolist(), cnt.tolist()):
            label_name = self.IDX2LABEL[u]
            counts[label_name] = c
        return counts


class INCARTDataset2Cls(BaseINCARTDataset):
    """
    Binary classification:
        Normal / Abnormal

    Mapping:
        Normal: N, L, R
        Abnormal: A, a, J, S, V, E, F, j, n, B
    Dropped:
        Q, /, f, +, [, ], p, t, ...
    """

    DATASET_NAME = "incart_2cls"

    SYMBOL2LABEL = {
        # Normal
        "N": "Normal",
        "L": "Normal",
        "R": "Normal",

        # Abnormal
        "A": "Abnormal",
        "a": "Abnormal",
        "J": "Abnormal",
        "S": "Abnormal",
        "V": "Abnormal",
        "E": "Abnormal",
        "F": "Abnormal",
        "j": "Abnormal",
        "n": "Abnormal",
        "B": "Abnormal",
    }

    LABEL2IDX = {
        "Normal": 0,
        "Abnormal": 1,
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


class INCARTDataset3Cls(BaseINCARTDataset):
    """
    AAMI-like 3-class classification:
        N / S / V

    Mapping:
        N: N, L, R, e, j, n, B
        S: A, a, J, S
        V: V, E, F   (F -> V)
    Dropped:
        Q, /, f, +, [, ], p, t, ...
    """

    DATASET_NAME = "incart_3cls"

    SYMBOL2LABEL = {
        # N
        "N": "N",
        "L": "N",
        "R": "N",
        "e": "N",
        "j": "N",
        "n": "N",
        "B": "N",

        # S
        "A": "S",
        "a": "S",
        "J": "S",
        "S": "S",

        # V
        "V": "V",
        "E": "V",
        "F": "V",   # F -> V
    }

    LABEL2IDX = {
        "N": 0,
        "S": 1,
        "V": 2,
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


class INCARTDataset5Cls(BaseINCARTDataset):
    """
    Top-5 frequent raw-class classification on INCART:
        N / V / R / A / F

    Based on your observed counts, the top-5 beat symbols are:
        N, V, R, A, F

    Dropped:
        all other symbols
    """

    DATASET_NAME = "incart_5cls"

    SYMBOL2LABEL = {
        "N": "N",
        "V": "V",
        "R": "R",
        "A": "A",
        "F": "F",
    }

    LABEL2IDX = {
        "N": 0,
        "V": 1,
        "R": 2,
        "A": 3,
        "F": 4,
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

class INCARTDatasetHF3Cls(BaseINCARTDataset):
    """
    High-frequency 3-class raw-label classification on INCART:
        N / V / A

    Based on the observed counts, the three selected raw beat symbols are:
        N, V, A

    Dropped:
        all other symbols
    """

    DATASET_NAME = "incart_hf3cls"

    SYMBOL2LABEL = {
        "N": "N",
        "V": "V",
        "A": "A",
    }

    LABEL2IDX = {
        "N": 0,
        "V": 1,
        "A": 2,
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

if __name__ == "__main__":
    root = "./datasets/incart"

    print("=" * 80)
    print("Testing INCART datasets")
    print("=" * 80)

    records = list_incart_records(Path(root))
    print(f"Found {len(records)} records")
    print(records[:5], "..." if len(records) > 5 else "")

    print("\n[2-class]")
    ds2 = INCARTDataset2Cls(root_dir=root, cache=False)
    print("Samples:", len(ds2))
    print("Class counts:", ds2.get_class_counts())
    x, y = ds2[0]
    print("One sample:", x.shape, y.item())

    print("\n[3-class]")
    ds3 = INCARTDataset3Cls(root_dir=root, cache=False)
    print("Samples:", len(ds3))
    print("Class counts:", ds3.get_class_counts())
    x, y = ds3[0]
    print("One sample:", x.shape, y.item())

    print("\n[5-class]")
    ds5 = INCARTDataset5Cls(root_dir=root, cache=False)
    print("Samples:", len(ds5))
    print("Class counts:", ds5.get_class_counts())
    x, y = ds5[0]
    print("One sample:", x.shape, y.item())

    print("\n[hf3-class]")
    ds_hf3 = INCARTDatasetHF3Cls(root_dir=root, cache=False)
    print("Samples:", len(ds_hf3))
    print("Class counts:", ds_hf3.get_class_counts())
    x, y = ds_hf3[0]
    print("One sample:", x.shape, y.item())