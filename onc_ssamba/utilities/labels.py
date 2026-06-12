"""Minimal label-format helpers for dataset tooling and notebooks."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List


def _normalize_labels(value: Any) -> List[str]:
    """Normalize a label field to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def load_labels(path: str) -> Dict[str, List[str]]:
    """Load labels JSON and normalize all values to list[str]."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        normalized[str(key)] = _normalize_labels(value)
    return normalized


def get_backward_compatible_labels(labels: Iterable[str]) -> List[str]:
    """
    Convert hierarchical labels to flat labels by taking the leaf segment.
    """
    flat: List[str] = []
    for label in labels:
        text = str(label)
        if " > " in text:
            flat.append(text.split(" > ")[-1])
        else:
            flat.append(text)
    return flat


def save_labels(
    output_file: str,
    label_data: Dict[str, Any],
    *,
    remove: bool = False,
    force_legacy_format: bool = False,
) -> None:
    """
    Merge labels into a JSON file.

    Args:
        output_file: Path to JSON file.
        label_data: Mapping filename -> label(s).
        remove: If True, remove provided labels.
        force_legacy_format: If True, flatten hierarchical labels on save.
    """
    current = load_labels(output_file) if os.path.exists(output_file) else {}

    for filename, labels in label_data.items():
        labels_list = _normalize_labels(labels)
        if remove:
            if filename in current:
                current[filename] = [l for l in current[filename] if l not in labels_list]
                if not current[filename]:
                    del current[filename]
        else:
            if filename in current:
                existing = set(current[filename])
                for label in labels_list:
                    if label not in existing:
                        current[filename].append(label)
                        existing.add(label)
            else:
                current[filename] = labels_list

    if force_legacy_format:
        current = {k: get_backward_compatible_labels(v) for k, v in current.items()}

    with open(output_file, "w") as f:
        json.dump(current, f, indent=4, sort_keys=True)
