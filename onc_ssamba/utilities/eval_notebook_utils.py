import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_per_class_probability_histograms(
	probs: np.ndarray,
	class_names: List[str],
	per_class_thresholds: Optional[np.ndarray] = None,
	bins: int = 50
) -> None:
	"""
	Plot histograms of predicted probabilities per class.
	If per_class_thresholds is provided (length C), add a vertical line per class.
	"""
	import math
	C = probs.shape[1]
	cols = min(4, C)
	rows = math.ceil(C / cols)
	plt.figure(figsize=(4 * cols, 3 * rows))
	for c in range(C):
		ax = plt.subplot(rows, cols, c + 1)
		vals = probs[:, c]
		ax.hist(vals, bins=bins, range=(0.0, 1.0), alpha=0.85, color='tab:blue')
		ax.set_title(class_names[c])
		ax.set_xlim(0.0, 1.0)
		if per_class_thresholds is not None and len(per_class_thresholds) == C:
			ax.axvline(float(per_class_thresholds[c]), color='red', linestyle='--', linewidth=1)
		ax.grid(alpha=0.2, linestyle='--')
	plt.suptitle('Predicted probability histograms per class', y=1.02)
	plt.tight_layout()
	plt.show()


