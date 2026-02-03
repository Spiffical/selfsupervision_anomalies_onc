from typing import List, Tuple, Optional, Callable
from pathlib import Path
import numpy as np
import pickle

try:
	import torch
	from torch.utils.data import Dataset, DataLoader  # type: ignore
except Exception:  # torch not always needed at import time
	torch = None  # type: ignore
 
from .spectrogram_utils import (
	load_mat_spectrogram,
	normalize_spectrogram,
	preprocess_with_resize_ctf,
	preprocess_to_tensor,
	resize_to_target,
)

# Optional tqdm import
try:
	from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
	def tqdm(x, **kwargs):  # type: ignore
		return x


TensorFn = Callable[[np.ndarray], 'torch.Tensor']
LoaderFn = Callable[[str, Tuple[int, int]], np.ndarray]
NormFn = Callable[[np.ndarray], np.ndarray]


def batched_inference(
	model,
	paths: List[str],
	loader_fn: LoaderFn,
	normalize_fn: NormFn,
	to_tensor_fn: TensorFn,
	expected_shape: Tuple[int, int],
	device: str = 'cuda',
	batch_size: int = 16,
	task: str = 'ft_cls',
	show_progress: bool = False,
	desc: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Generic batched inference for .mat spectrogram files.
	- loader_fn(path, expected_shape) -> np.ndarray [F, T]
	- normalize_fn(np.ndarray) -> np.ndarray [F, T]
	- to_tensor_fn(np.ndarray) -> torch.Tensor [C, T, F] or [C, F, T] depending on model expectation
	Returns (logits, probs, argmax_preds)
	"""
	if torch is None:
		raise RuntimeError("Torch is required for batched_inference but is not available")
	logits_list = []
	model.eval()
	with torch.no_grad():
		iterator = range(0, len(paths), batch_size)
		if show_progress:
			iterator = tqdm(iterator, desc=desc or 'inference', unit='batch')
		for i in iterator:
			batch_paths = paths[i:i+batch_size]
			batch_tensors = []
			for p in batch_paths:
				arr = loader_fn(p, expected_shape)
				if arr is None:
					continue
				arr = normalize_fn(arr)
				ten = to_tensor_fn(arr)
				batch_tensors.append(ten)
			if not batch_tensors:
				continue
			x = torch.stack(batch_tensors, dim=0).to(device)
			out = model(x, task=task)
			logits_list.append(out.detach().cpu().numpy())
	logits = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 0), dtype=np.float32)
	if logits.size == 0:
		probs = np.zeros_like(logits)
		preds = np.array([], dtype=int)
	else:
		probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
		preds = probs.argmax(axis=1)
	return logits, probs, preds


def topk_from_probs(probs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Return (indices, values) of top-k per row from probs array.
	"""
	order = np.argsort(-probs, axis=1)
	idx = order[:, :k]
	vals = probs[np.arange(len(probs))[:, None], idx]
	return idx, vals


# ---------------------------------------------------------
# Standardized dataloader + inference used across notebooks
# ---------------------------------------------------------

class MatSpectrogramDataset(Dataset):
	"""
	Dataset for unlabeled .mat spectrogram files returning (tensor, label=-1, source).
	- Reads using load_mat_spectrogram
	- Normalizes using normalize_spectrogram with optional dataset stats
	- Resizes to target and converts to tensor [1, T, F] via preprocess_with_resize_ctf
	"""

	def __init__(
		self,
		paths: List[str],
		expected_shape: Tuple[int, int],
		target_size: Tuple[int, int],
		dataset_mean: Optional[float] = None,
		dataset_std: Optional[float] = None,
		amount: float = 1.0,
	):
		if torch is None:
			raise RuntimeError("Torch is required for MatSpectrogramDataset but is not available")
		self.paths = paths
		self.expected_shape = expected_shape
		self.target_size = target_size
		self.dataset_mean = dataset_mean
		self.dataset_std = dataset_std
		self.amount = amount

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int):
		path = self.paths[index]
		arr = load_mat_spectrogram(path, self.expected_shape)
		arr = normalize_spectrogram(arr, dataset_mean=self.dataset_mean, dataset_std=self.dataset_std, amount=self.amount)
		tensor = preprocess_with_resize_ctf(arr, self.target_size)
		# Use filename stem as source identifier (or directory name if needed)
		source = None
		try:
			import os
			source = os.path.basename(path).split('_')[0]
		except Exception:
			source = None
		return tensor, -1, source


def build_mat_dataloader(
	paths: List[str],
	expected_shape: Tuple[int, int],
	target_size: Tuple[int, int],
	batch_size: int = 16,
	num_workers: int = 0,
	dataset_mean: Optional[float] = None,
	dataset_std: Optional[float] = None,
	amount: float = 1.0,
):
	"""
	Create a DataLoader over .mat spectrogram files, standardized to model input.
	"""
	if torch is None:
		raise RuntimeError("Torch is required for build_mat_dataloader but is not available")
	ds = MatSpectrogramDataset(
		paths=paths,
		expected_shape=expected_shape,
		target_size=target_size,
		dataset_mean=dataset_mean,
		dataset_std=dataset_std,
		amount=amount,
	)
	return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------
# H5-like pipeline: resize -> normalize -> [C, F, T]
# ---------------------------------------------------------

class MatSpectrogramDatasetH5Like(Dataset):
	"""
	Dataset for .mat spectrogram files using H5-like preprocessing order:
	- Resize to target size first
	- Normalize using dataset statistics or percentile-based method
	- Convert to tensor shape [C, F, T] via preprocess_to_tensor
	"""

	def __init__(
		self,
		paths: List[str],
		expected_shape: Tuple[int, int],
		target_size: Tuple[int, int],
		dataset_mean: Optional[float] = None,
		dataset_std: Optional[float] = None,
		amount: float = 1.0,
	):
		if torch is None:
			raise RuntimeError("Torch is required for MatSpectrogramDatasetH5Like but is not available")
		self.paths = paths
		self.expected_shape = expected_shape
		self.target_size = target_size
		self.dataset_mean = dataset_mean
		self.dataset_std = dataset_std
		self.amount = amount

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int):
		path = self.paths[index]
		arr = load_mat_spectrogram(path, self.expected_shape)
		# H5-like order: resize -> normalize
		arr = resize_to_target(arr, self.target_size)
		arr = normalize_spectrogram(
			arr,
			dataset_mean=self.dataset_mean,
			dataset_std=self.dataset_std,
			amount=self.amount,
		)
		tensor = preprocess_to_tensor(arr)  # [C, F, T]
		# Optional source id from filename prefix
		try:
			import os
			source = os.path.basename(path).split('_')[0]
		except Exception:
			source = None
		return tensor, -1, source


def build_mat_dataloader_h5like(
	paths: List[str],
	expected_shape: Tuple[int, int],
	target_size: Tuple[int, int],
	batch_size: int = 16,
	num_workers: int = 0,
	dataset_mean: Optional[float] = None,
	dataset_std: Optional[float] = None,
	amount: float = 1.0,
):
	"""
	Create a DataLoader over .mat files using H5-like preprocessing (resize -> normalize -> [C,F,T]).
	"""
	if torch is None:
		raise RuntimeError("Torch is required for build_mat_dataloader_h5like but is not available")
	ds = MatSpectrogramDatasetH5Like(
		paths=paths,
		expected_shape=expected_shape,
		target_size=target_size,
		dataset_mean=dataset_mean,
		dataset_std=dataset_std,
		amount=amount,
	)
	return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_inference_multiclass(model, data_loader, device, task: str = 'ft_cls', show_progress: bool = False, desc: Optional[str] = None):
	"""
	Standard multiclass inference using a DataLoader that yields (data, label, source).
	- Accumulates logits, converts to probabilities with softmax.
	- Returns (y_true, y_pred, y_proba, sources). y_true will be -1 for unlabeled datasets.
	"""
	if torch is None:
		raise RuntimeError("Torch is required for run_inference_multiclass but is not available")
	model.eval()
	all_logits: List[torch.Tensor] = []
	all_targets: List[torch.Tensor] = []
	all_sources: List[Optional[str]] = []
	with torch.no_grad():
		iterator = data_loader
		if show_progress:
			iterator = tqdm(data_loader, desc=desc or 'inference', unit='batch')
		for batch in iterator:
			# Support datasets that return (data, label, source) or (data, label)
			if isinstance(batch, (list, tuple)) and len(batch) == 3:
				data, labels, source = batch
				all_sources.extend(list(source))
			elif isinstance(batch, (list, tuple)) and len(batch) == 2:
				data, labels = batch
				all_sources.extend([None] * len(labels))
			else:
				data = batch
				labels = torch.full((data.shape[0],), -1, dtype=torch.long)
				all_sources.extend([None] * data.shape[0])
			data = data.to(device)
			logits = model(data, task=task)
			if logits.dim() == 1:
				logits = logits.unsqueeze(1)
			all_logits.append(logits.cpu())
			all_targets.append(labels.long().cpu())
	logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 0))
	targets_cat = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0,), dtype=torch.long)
	if logits_cat.numel() == 0:
		probs = np.zeros((0, 0), dtype=np.float32)
		preds = np.array([], dtype=int)
	else:
		probs_t = torch.softmax(logits_cat, dim=1)
		probs = probs_t.numpy()
		preds = probs.argmax(axis=1)
	return targets_cat.numpy(), preds, probs, all_sources
# ---------------------------------------------------------
# Simple finetune inference helpers
# ---------------------------------------------------------

def resolve_finetuned_checkpoint(
	model_dir,
	checkpoint_path: Optional[str] = None,
	task: Optional[str] = None,
):
	"""
	Resolve a finetuned checkpoint path.
	- If checkpoint_path is provided, validates it exists.
	- Otherwise, looks for common *best_checkpoint.pth patterns.
	"""
	model_dir = Path(model_dir)
	if checkpoint_path:
		ckpt = Path(checkpoint_path)
		if not ckpt.exists():
			raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
		return ckpt
	models_dir = model_dir / 'models'
	candidates: List[Path] = []
	if task:
		candidates.append(models_dir / f"{task.replace('_','-')}_best_checkpoint.pth")
	candidates.extend([
		models_dir / 'ft-avgtok_best_checkpoint.pth',
		models_dir / 'ft-cls_best_checkpoint.pth',
		models_dir / 'best_checkpoint.pth',
		model_dir / 'ft-avgtok_best_checkpoint.pth',
		model_dir / 'ft-cls_best_checkpoint.pth',
		model_dir / 'best_checkpoint.pth',
	])
	for p in candidates:
		if p.exists():
			return p
	# Fallback: any best checkpoint, prefer most recently modified
	best = []
	if models_dir.exists():
		best.extend(list(models_dir.glob('*best_checkpoint.pth')))
	best.extend(list(model_dir.glob('*best_checkpoint.pth')))
	if best:
		best.sort(key=lambda x: x.stat().st_mtime, reverse=True)
		return best[0]
	# Fallback: latest checkpoint (not necessarily best)
	try:
		from .checkpoint_utils import find_latest_checkpoint
		_, latest = find_latest_checkpoint(str(model_dir), task=task)
		if latest:
			return Path(latest)
	except Exception:
		pass
	raise FileNotFoundError(f"No checkpoint found in {model_dir}")

def load_finetuned_model(
	model_dir,
	checkpoint_path: Optional[str] = None,
	device=None,
	multiclass: bool = True,
	task: Optional[str] = None,
):
	"""
	Load a finetuned model + args from a model directory.
	Returns (model, args, checkpoint_path, checkpoint_dict).
	"""
	if torch is None:
		raise RuntimeError("Torch is required for load_finetuned_model but is not available")
	model_dir = Path(model_dir)
	args_path = model_dir / 'args.pkl'
	if not args_path.exists():
		raise FileNotFoundError(f"args.pkl not found in {model_dir}")
	with args_path.open('rb') as f:
		args = pickle.load(f)
	if multiclass:
		args.multiclass = True
	if task:
		args.task = task
	args.exp_dir = model_dir

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	from .training_utils import create_model
	model = create_model(args).to(device)

	ckpt_path = resolve_finetuned_checkpoint(model_dir, checkpoint_path, task=getattr(args, 'task', None))
	from .checkpoint_utils import load_checkpoint
	checkpoint = load_checkpoint(str(ckpt_path), device)
	state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

	if len(state_dict) > 0:
		if isinstance(model, torch.nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
			state_dict = {f"module.{k}": v for k, v in state_dict.items()}
		elif (not isinstance(model, torch.nn.DataParallel)) and list(state_dict.keys())[0].startswith('module.'):
			state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

	model.load_state_dict(state_dict)
	model.eval()
	return model, args, ckpt_path, checkpoint

def predict_mat_paths(
	model,
	paths: List[str],
	args=None,
	expected_shape: Tuple[int, int] = (854, 1000),
	target_size: Optional[Tuple[int, int]] = None,
	dataset_mean: Optional[float] = None,
	dataset_std: Optional[float] = None,
	amount: Optional[float] = None,
	batch_size: int = 32,
	num_workers: int = 0,
	device=None,
	task: Optional[str] = None,
	h5_like: bool = True,
	show_progress: bool = False,
	desc: Optional[str] = None,
):
	"""
	Run inference on a list of .mat paths.
	Returns (y_true, preds, probs, sources) from run_inference_multiclass.
	"""
	if torch is None:
		raise RuntimeError("Torch is required for predict_mat_paths but is not available")
	paths = [str(p) for p in paths]
	if args is not None:
		if target_size is None:
			target_size = (getattr(args, 'num_mel_bins', 128), getattr(args, 'target_length', 1024))
		if dataset_mean is None:
			dataset_mean = getattr(args, 'dataset_mean', None)
		if dataset_std is None:
			dataset_std = getattr(args, 'dataset_std', None)
		if amount is None:
			amount = getattr(args, 'amount', 1.0)
		if task is None:
			task = getattr(args, 'task', 'ft_cls')
	if target_size is None:
		target_size = (128, 1024)
	if amount is None:
		amount = 1.0
	if device is None:
		try:
			device = next(model.parameters()).device
		except Exception:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if h5_like:
		data_loader = build_mat_dataloader_h5like(
			paths=paths,
			expected_shape=expected_shape,
			target_size=target_size,
			batch_size=batch_size,
			num_workers=num_workers,
			dataset_mean=dataset_mean,
			dataset_std=dataset_std,
			amount=amount,
		)
	else:
		data_loader = build_mat_dataloader(
			paths=paths,
			expected_shape=expected_shape,
			target_size=target_size,
			batch_size=batch_size,
			num_workers=num_workers,
			dataset_mean=dataset_mean,
			dataset_std=dataset_std,
			amount=amount,
		)

	return run_inference_multiclass(
		model=model,
		data_loader=data_loader,
		device=device,
		task=task or 'ft_cls',
		show_progress=show_progress,
		desc=desc,
	)

def predict_mat_dir(
	model,
	mat_dir,
	args=None,
	pattern: str = '**/*.mat',
	**kwargs,
):
	"""
	Convenience wrapper for running inference on a directory of .mat files.
	Returns (paths, y_true, preds, probs, sources).
	"""
	mat_dir = Path(mat_dir)
	paths = sorted([str(p) for p in mat_dir.glob(pattern)])
	y_true, preds, probs, sources = predict_mat_paths(
		model=model,
		paths=paths,
		args=args,
		**kwargs,
	)
	return paths, y_true, preds, probs, sources
