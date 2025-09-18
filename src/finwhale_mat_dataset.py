import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import scipy.io as sio
except Exception as e:
    sio = None  # will raise at runtime if used without SciPy


SPECTRO_KEYS: Sequence[str] = (
    'spectrogram', 'PdB_norm', 'power_db_norm', 'PdB', 'P_db',
    'P', 'PSD', 'psd', 'Sxx', 'S', 'spec', 'power_spectrogram'
)
FREQ_KEYS: Sequence[str] = ('frequencies', 'F', 'freqs', 'freq', 'f')
TIME_KEYS: Sequence[str] = ('times', 'T', 'time', 't')


def _list_mat_files(folder: Path) -> List[Path]:
    out: List[Path] = []
    # Use scandir for performance on huge dirs
    for entry in os.scandir(folder):
        try:
            if entry.is_file() and entry.name.lower().endswith('.mat'):
                out.append(Path(entry.path))
        except FileNotFoundError:
            continue
    out.sort()
    return out


def _find_key(d: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    # Case-insensitive fallback
    lowered = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in lowered:
            return lowered[k.lower()]
    return None


def _normalize_db_to_unit(x: np.ndarray, min_db: float = -80.0, max_db: float = 0.0) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.clip(x, min_db, max_db)
    return (x - min_db) / (max_db - min_db)


def _start_from_fraction(T: int, crop: int, frac_in_crop: float) -> int:
    """Compute crop start so that the spectrogram center (call) appears at a given
    fraction position inside the crop (0=left edge, 1=right edge)."""
    frac = float(np.clip(frac_in_crop, 0.0, 1.0))
    call_center = T // 2
    pos_in_crop = frac * max(1, crop - 1)
    start = int(round(call_center - pos_in_crop))
    return max(0, min(start, max(0, T - crop)))


def _choose_start_idx(T: int, crop: int, split: str, is_positive: bool,
                      center_bias_sigma_frac: float = 0.25,
                      rng: Optional[np.random.Generator] = None,
                      augment_eval: bool = False) -> int:
    if T <= crop:
        return 0
    if rng is None:
        rng = np.random.default_rng()
    if is_positive:
        # Keep the call in view by selecting a fraction position inside the crop.
        if split == 'train' or (split != 'train' and augment_eval):
            # Gaussian jitter around center (0.5), sigma is fraction of half-range
            sigma = max(1e-3, float(center_bias_sigma_frac)) * 0.5
            frac = None
            # Truncate to [0,1]
            for _ in range(10):
                f = 0.5 + float(rng.normal(0.0, sigma))
                if 0.0 <= f <= 1.0:
                    frac = f
                    break
            if frac is None:
                frac = 0.5
            return _start_from_fraction(T, crop, frac)
        else:
            # Deterministic center for val/test without augmentation
            return _start_from_fraction(T, crop, 0.5)
    else:
        # Negatives: uniform random window for train; deterministic center for eval
        if split == 'train' or (split != 'train' and augment_eval):
            return int(rng.integers(0, T - crop + 1))
        else:
            center = T // 2
            return max(0, min(center - crop // 2, T - crop))


class FinWhaleMatDataset(Dataset):
    """Supervised spectrogram dataset from MAT files.

    - Positive samples from `pos_dir` (call present)
    - Negative samples from `neg_dir` (no call)
    - Applies normalization to [0, 1] from dB values and square crop on time axis
    - Training-time augmentation: random in-crop position for positives; random window for negatives
    """

    def __init__(
        self,
        pos_dir: str,
        neg_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        crop_size: int = 96,
        min_db: float = -80.0,
        max_db: float = 0.0,
        center_bias_sigma_frac: float = 0.25,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_path: bool = False,
        seed: int = 0,
        augment_eval: bool = False,
        return_meta: bool = False,
        file_list: Optional[List[Tuple[Path, int]]] = None,
    ) -> None:
        if sio is None:
            raise RuntimeError("scipy is required to load .mat files. Please install scipy.")
        self.pos_dir = Path(pos_dir)
        self.neg_dir = Path(neg_dir)
        self.split = split
        self.crop_size = int(crop_size)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.center_bias_sigma_frac = float(center_bias_sigma_frac)
        self.transform = transform
        self.return_path = return_path
        self.rng = np.random.default_rng(seed)
        self.augment_eval = bool(augment_eval)
        self.return_meta = bool(return_meta)

        if file_list is not None:
            # Use provided list directly
            self.files = [(Path(p), int(lbl)) for p, lbl in file_list]
            return

        if not self.pos_dir.exists():
            raise FileNotFoundError(f"Positive directory not found: {self.pos_dir}")
        if not self.neg_dir.exists():
            raise FileNotFoundError(f"Negative directory not found: {self.neg_dir}")

        self.pos_files = _list_mat_files(self.pos_dir)
        self.neg_files = _list_mat_files(self.neg_dir)

        # Stratified split per class
        def split_files(files: List[Path]) -> List[Path]:
            n = len(files)
            idx = np.arange(n)
            # Deterministic shuffle based on seed
            rng_local = np.random.default_rng(seed)
            rng_local.shuffle(idx)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            if split == 'train':
                sel = idx[:n_train]
            elif split == 'val':
                sel = idx[n_train:n_train + n_val]
            else:
                sel = idx[n_train + n_val:]
            return [files[i] for i in sel]

        pos_sel = split_files(self.pos_files)
        neg_sel = split_files(self.neg_files)

        # Merge lists and create labels (1 for pos, 0 for neg)
        self.files: List[Tuple[Path, int]] = [(p, 1) for p in pos_sel] + [(n, 0) for n in neg_sel]

    def __len__(self) -> int:
        return len(self.files)

    def _load_spectrogram(self, path: Path) -> np.ndarray:
        # Load and pick key
        data = sio.loadmat(str(path), simplify_cells=True)
        k = _find_key(data, SPECTRO_KEYS)
        if k is None:
            raise KeyError(f"No spectrogram-like key found in {path.name}")
        spec = np.asarray(data[k])
        if spec.ndim != 2:
            raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {path.name}")
        # Attempt to detect (F, T) orientation; default is as-is
        # If available, align using freq/time vectors
        fk = _find_key(data, FREQ_KEYS)
        tk = _find_key(data, TIME_KEYS)
        if fk in data and tk in data:
            f_len = int(np.asarray(data[fk]).ravel().shape[0])
            t_len = int(np.asarray(data[tk]).ravel().shape[0])
            r, c = spec.shape[:2]
            if (r, c) == (t_len, f_len):
                spec = spec.T  # now (F, T)
        return spec

    def _crop(self, spec: np.ndarray, is_positive: bool, start_override: Optional[int] = None) -> np.ndarray:
        # spec expected (F, T). Crop to (F, crop_size) then square to (crop_size, crop_size)
        F, T = spec.shape
        # If freq dimension not equal to crop_size, pad or crop frequency axis to crop_size
        f_start = 0
        if F < self.crop_size:
            # pad on bottom with min value
            pad = self.crop_size - F
            spec = np.pad(spec, ((0, pad), (0, 0)), mode='edge')
            F = self.crop_size
        elif F > self.crop_size:
            # center-crop frequency
            f_start = max(0, (F - self.crop_size) // 2)
            spec = spec[f_start:f_start + self.crop_size, :]
            F = self.crop_size

        # Time crop to crop_size
        if start_override is not None:
            start = int(start_override)
        else:
            split_for_crop = self.split
            start = _choose_start_idx(T, self.crop_size, split_for_crop, is_positive,
                                      center_bias_sigma_frac=self.center_bias_sigma_frac,
                                      rng=self.rng,
                                      augment_eval=self.augment_eval)
        end = start + self.crop_size
        if T < self.crop_size:
            # pad time with edge
            pad = self.crop_size - T
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='edge')
        else:
            spec = spec[:, start:end]
        return spec

    def __getitem__(self, index: int):
        path, label = self.files[index]
        spec = self._load_spectrogram(path)
        F, T = spec.shape
        # Normalize to [0,1]
        spec = _normalize_db_to_unit(spec, self.min_db, self.max_db)
        # Determine crop start ensuring call visibility for positives
        start = _choose_start_idx(T, self.crop_size, self.split, bool(label),
                                  center_bias_sigma_frac=self.center_bias_sigma_frac,
                                  rng=self.rng,
                                  augment_eval=self.augment_eval)
        # Crop/augment
        spec = self._crop(spec, is_positive=bool(label), start_override=start)
        # To torch [C=1, F, T]
        x = torch.from_numpy(spec).unsqueeze(0).float()
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(label, dtype=torch.long)
        if self.return_meta or self.return_path:
            meta = None
            if self.return_meta:
                # Distance of crop center from spectrogram center (frames and fraction)
                crop_center = start + (self.crop_size // 2)
                spec_center = T // 2
                dist_frames = abs(crop_center - spec_center)
                max_center = max(1, T // 2)
                dist_frac = float(dist_frames) / float(max_center)
                meta = {
                    'crop_start': int(start),
                    'full_T': int(T),
                    'crop_size': int(self.crop_size),
                    'dist_from_center_frames': int(dist_frames),
                    'dist_from_center_frac': float(dist_frac),
                }
            if self.return_meta and self.return_path:
                return x, y, str(path), meta
            if self.return_path:
                return x, y, str(path)
            if self.return_meta:
                return x, y, meta
        return x, y


def make_dataloaders(
    pos_dir: str,
    neg_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    crop_size: int = 96,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_db: float = -80.0,
    max_db: float = 0.0,
    center_bias_sigma_frac: float = 0.25,
    seed: int = 0,
    balance: str = 'weighted',  # 'weighted' | 'oversample' | 'none'
    augment_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with optional class balancing for train.

    Splits are stratified by listing order (deterministic) with an 80/10/10 split.
    For different splits, build datasets manually.
    """
    train_ds = FinWhaleMatDataset(
        pos_dir, neg_dir, split='train', train_ratio=train_ratio, val_ratio=val_ratio, crop_size=crop_size,
        min_db=min_db, max_db=max_db, center_bias_sigma_frac=center_bias_sigma_frac,
        seed=seed
    )
    val_ds = FinWhaleMatDataset(
        pos_dir, neg_dir, split='val', train_ratio=train_ratio, val_ratio=val_ratio, crop_size=crop_size,
        min_db=min_db, max_db=max_db, center_bias_sigma_frac=center_bias_sigma_frac,
        seed=seed
    )
    test_ds = FinWhaleMatDataset(
        pos_dir, neg_dir, split='test', train_ratio=train_ratio, val_ratio=val_ratio, crop_size=crop_size,
        min_db=min_db, max_db=max_db, center_bias_sigma_frac=center_bias_sigma_frac,
        seed=seed, augment_eval=augment_test
    )

    # Build train sampler for balancing
    sampler = None
    if balance in ('weighted', 'oversample'):
        # Compute per-sample weights inversely to class frequency
        labels = torch.tensor([lbl for _, lbl in train_ds.files], dtype=torch.long)
        class_counts = torch.bincount(labels, minlength=2).float()
        class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
        sample_weights = class_weights[labels]
        if balance == 'weighted':
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
        else:  # oversample
            # Target equal class count per epoch = 2 * max(counts)
            target = int(2 * torch.max(class_counts).item())
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=target, replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
