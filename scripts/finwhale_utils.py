#!/usr/bin/env python3
import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Iterator

FILENAME_RE_COMMA = re.compile(r"^(?P<id>[^,]+),(?P<start>[\d\.]+)s,(?P<dur>[\d\.]+)s(?:,.*)?$")
FILENAME_RE_UNDERSCORE = re.compile(r"^(?P<id>.+?)_(?P<start>[\d\.]+)s_(?P<dur>[\d\.]+)s(?:_.*)?$")


def parse_mat_filename(filename: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Parse a MAT filename to extract source audio id and time info.

    Supports:
    - Underscore form: ICLISTENHF1353_20180705T050128.948Z_104.8s_105.3s_40Hz_custom.mat
    - Comma form:      ICLISTENHF1353_20180705T050128.948Z,104.8s,105.3s,_40Hz_custom.mat

    Returns: (source_id, start_seconds, duration_seconds)
    If time parsing fails, returns None for those fields.
    """
    name = filename
    if name.lower().endswith('.mat'):
        name = name[:-4]

    m = FILENAME_RE_UNDERSCORE.match(name)
    if not m:
        m = FILENAME_RE_COMMA.match(name)

    if not m:
        # Fallback: return the stem as id
        stem = Path(filename).stem
        return stem, None, None

    src_id = m.group('id')
    try:
        start_s = float(m.group('start'))
    except Exception:
        start_s = None
    try:
        dur_s = float(m.group('dur'))
    except Exception:
        dur_s = None
    return src_id, start_s, dur_s


def list_mat_files(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() == '.mat'])


def iter_mat_files(dir_path: str) -> Iterator[Path]:
    """Efficiently iterate .mat files in a directory using os.scandir without building a huge list."""
    for entry in os.scandir(dir_path):
        try:
            if entry.is_file() and entry.name.lower().endswith('.mat'):
                yield Path(entry.path)
        except FileNotFoundError:
            continue
