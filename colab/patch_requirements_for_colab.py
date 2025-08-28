#!/usr/bin/env python3
"""
Patch repository requirements to be Colab-friendly:
- Comment out strict pins that conflict with Colab runtime (e.g., ipython).
- Ensure no protobuf pin in mamba requirements.

Usage:
  python colab/patch_requirements_for_colab.py [repo_root]
"""
import sys
from pathlib import Path

def comment_pin(path: Path, prefixes):
    if not path.exists():
        return False
    text = path.read_text()
    out_lines = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.lower().startswith(p.lower() + '==') for p in prefixes):
            out_lines.append(f"# [colab-patched] {line}")
            changed = True
        else:
            out_lines.append(line)
    if changed:
        path.write_text("\n".join(out_lines) + "\n")
    return changed

def main():
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    base = repo_root / 'requirements-base.txt'
    mamba = repo_root / 'requirements-mamba.txt'

    ch1 = comment_pin(base, ['ipython'])
    # In case protobuf shows up in mamba requirements in the future
    ch2 = comment_pin(mamba, ['protobuf'])
    if ch1 or ch2:
        print('Patched requirements to avoid Colab conflicts:', {'base': ch1, 'mamba': ch2})
    else:
        print('No conflicting pins found to patch.')

if __name__ == '__main__':
    main()

