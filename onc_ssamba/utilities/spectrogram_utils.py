import numpy as np
from typing import Optional, Tuple
import scipy.io as sio
import numpy.typing as npt

try:
	import torch
except Exception:  # torch not always needed at import time
	torch = None  # type: ignore

try:
	import cv2  # type: ignore
except Exception:
	cv2 = None  # type: ignore


def load_mat_spectrogram(mat_file_path: str, expected_shape: Tuple[int, int]) -> npt.NDArray[np.float32]:
	"""
	Load a spectrogram from a MATLAB .mat file and return as float32 array [F, T].
	Pads in time dimension if shorter than expected_shape[1].
	"""
	mat_data = sio.loadmat(mat_file_path)
	if 'SpectData' not in mat_data:
		raise ValueError(f"'SpectData' not found in {mat_file_path}")
	data = mat_data['SpectData']['PSD'][0, 0]
	if data.shape[1] < expected_shape[1]:
		padding_width = ((0, 0), (0, expected_shape[1] - data.shape[1]))
		data = np.pad(data, padding_width, mode='constant', constant_values=0)
	data[np.isinf(data)] = 0
	data = np.nan_to_num(data, 0).astype(np.float32)
	return data


def normalize_spectrogram(
	data: npt.NDArray[np.float32],
	dataset_mean: Optional[float] = None,
	dataset_std: Optional[float] = None,
	amount: float = 1.0,
) -> npt.NDArray[np.float32]:
	"""
	Normalize spectrogram values.
	- If dataset_mean and dataset_std provided: (x - mean) / (2*std)
	- Else: percentile clip [amount, 100-amount], log, min-max to [0, 1]
	"""
	data = data.astype(np.float32)
	if dataset_mean is not None and dataset_std is not None:
		denom = float(dataset_std) * 2.0 if float(dataset_std) != 0 else 1.0
		data = (data - float(dataset_mean)) / denom
	else:
		low, high = np.percentile(data, [amount, 100 - amount])
		data = np.clip(data, low, high)
		data = np.log(data + 1e-8)
		min_v = np.min(data)
		max_v = np.max(data)
		denom = (max_v - min_v) + 1e-8
		data = (data - min_v) / denom
	return np.nan_to_num(data, 0).astype(np.float32)


def resize_to_target(data: npt.NDArray[np.float32], target_size: Tuple[int, int]) -> npt.NDArray[np.float32]:
	"""
	Resize [F, T] to [target_F, target_T] using area interpolation if available.
	target_size is (target_F, target_T).
	"""
	if cv2 is None:
		# Fallback: simple numpy zoom via slicing/padding (not ideal). Here, just center-crop or pad.
		F, T = data.shape
		tF, tT = target_size
		# Pad
		padF = max(0, tF - F)
		padT = max(0, tT - T)
		if padF > 0 or padT > 0:
			data = np.pad(data, ((padF // 2, padF - padF // 2), (padT // 2, padT - padT // 2)), mode='constant')
		# Crop to target
		F, T = data.shape
		startF = max(0, (F - tF) // 2)
		startT = max(0, (T - tT) // 2)
		return data[startF:startF + tF, startT:startT + tT].astype(np.float32)
	# OpenCV expects (width, height) -> (T, F)
	resized = cv2.resize(data, (int(target_size[1]), int(target_size[0])))
	return resized.astype(np.float32)


def preprocess_to_tensor(data: npt.NDArray[np.float32]):
	"""
	Convert [F, T] array to torch tensor [C, F, T] with C=1.
	"""
	if torch is None:
		raise RuntimeError("Torch is required for preprocess_to_tensor but is not available")
	data = np.expand_dims(data, axis=-1)
	data = np.transpose(data, (2, 0, 1))
	return torch.from_numpy(data)


def preprocess_to_tensor_ctf(data: npt.NDArray[np.float32]):
	"""
	Convert [F, T] array to torch tensor [C, T, F] with C=1 (model expects [B,C,T,F]).
	"""
	if torch is None:
		raise RuntimeError("Torch is required for preprocess_to_tensor_ctf but is not available")
	# Move time to axis 1, freq to axis 2
	data = data.T  # [T, F]
	data = np.expand_dims(data, axis=0)  # [1, T, F]
	return torch.from_numpy(data.astype(np.float32))


def preprocess_with_resize_ctf(data: npt.NDArray[np.float32], target_size: Tuple[int, int]):
	"""
	Resize [F, T] to target and return tensor [1, T, F].
	"""
	res = resize_to_target(data, target_size)
	return preprocess_to_tensor_ctf(res)
