from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def show_spectrogram_grid(paths: List[str],
                         loader_fn,
                         normalize_fn,
                         expected_shape,
                         title: str,
                         k: int = 8,
                         samples_per_row: int = 4,
                         cmap: str = 'inferno') -> None:
	idxs = list(range(min(k, len(paths))))
	if not idxs:
		print(f"No examples for {title}")
		return
	rows = int(np.ceil(len(idxs) / samples_per_row))
	plt.figure(figsize=(5 * samples_per_row, 3 * rows))
	plt.suptitle(title, fontsize=14)
	for j, i in enumerate(idxs):
		arr = loader_fn(paths[i], expected_shape)
		if arr is None:
			continue
		arr = normalize_fn(arr)
		arr = np.expand_dims(arr, axis=-1)
		arr = np.transpose(arr, (2, 0, 1))
		plt.subplot(rows, samples_per_row, j + 1)
		plt.imshow(arr[0], origin='lower', aspect='auto', cmap=cmap)
		plt.xlabel(Path(paths[i]).name, fontsize=7)
		plt.axis('off')
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()


