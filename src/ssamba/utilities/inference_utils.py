from typing import List, Tuple, Optional, Callable
import numpy as np

try:
	import torch
except Exception:  # torch not always needed at import time
	torch = None  # type: ignore


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
	task: str = 'ft_cls'
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
		for i in range(0, len(paths), batch_size):
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
