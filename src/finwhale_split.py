import random
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.finwhale_utils import iter_mat_files, parse_mat_filename  # reuse


def build_entries(pos_dir: str, neg_dir: str) -> List[dict]:
    entries: List[dict] = []
    for p in iter_mat_files(pos_dir):
        src, start, dur = parse_mat_filename(Path(p).name)
        entries.append({'path': Path(p), 'src': src, 'start': start, 'dur': dur, 'label': 1})
    for p in iter_mat_files(neg_dir):
        src, start, dur = parse_mat_filename(Path(p).name)
        entries.append({'path': Path(p), 'src': src, 'start': start, 'dur': dur, 'label': 0})
    return entries


def split_group_by_source(entries: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for e in entries:
        groups.setdefault(e['src'], []).append(e)
    src_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(src_ids)
    n = len(src_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split_srcs = {
        'train': src_ids[:n_train],
        'val': src_ids[n_train:n_train + n_val],
        'test': src_ids[n_train + n_val:]
    }
    return {k: sum([groups[sid] for sid in v], []) for k, v in split_srcs.items()}


def split_time_separated(entries: List[dict], train_ratio: float, val_ratio: float, seed: int, min_gap_seconds: float) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    entries = list(entries)
    rng.shuffle(entries)

    split_map: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}
    n = len(entries)
    t_train = int(n * train_ratio)
    t_val = int(n * val_ratio)

    def violates(target: str, e: dict) -> bool:
        if e['start'] is None:
            return False
        center = float(e['start']) + float(e['dur'] or 0.0) * 0.5
        src = e['src']
        for split, items in split_map.items():
            if split == target:
                continue
            for it in items:
                if it['src'] != src or it['start'] is None:
                    continue
                oc = float(it['start']) + float(it['dur'] or 0.0) * 0.5
                if abs(center - oc) < min_gap_seconds:
                    return True
        return False

    for e in entries:
        order = ['train', 'val', 'test']
        order.sort(key=lambda s: (len(split_map[s]) / max(1, {'train': t_train, 'val': t_val, 'test': n - t_train - t_val}[s])))
        placed = False
        for s in order:
            if not violates(s, e):
                split_map[s].append(e)
                placed = True
                break
        if not placed:
            sizes = sorted(['train', 'val', 'test'], key=lambda s: len(split_map[s]))
            split_map[sizes[0]].append(e)
    return split_map


def summarise_counts(split: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, lst in split.items():
        pos = sum(1 for e in lst if e['label'] == 1)
        neg = sum(1 for e in lst if e['label'] == 0)
        out[k] = {'pos': pos, 'neg': neg, 'total': len(lst)}
    return out
