#!/usr/bin/env python3
"""
generate_requirements.py

Generate a filtered `requirements.txt` from `pip freeze`, excluding packages
like `torch` (useful when you install PyTorch from the CUDA wheel separately).

Usage examples:
  python generate_requirements.py --output requirements.txt --append-torch
  python generate_requirements.py --from-file all-requirements.txt --exclude '^torch' --output reqs.txt

Options:
  --output: output file (default: requirements.txt)
  --from-file: read frozen requirements from this file instead of running `pip freeze`
  --exclude: regex pattern to exclude (can be repeated). Default excludes torch, torchvision, torchaudio, torchtext.
  --append-torch: append a placeholder comment with pip install instructions for the CUDA wheel
  --torch-instruction: custom instruction to append
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXCLUDES = [
    r'^(torch|torchvision|torchaudio|torchtext)\b',
    r'^(conda|anaconda|pkg-resources)\b',
    r'^file:',
    r'^git\+',
    r'^-e\b',
]


def parse_args():
    p = argparse.ArgumentParser(description='Generate filtered requirements.txt excluding torch packages')
    p.add_argument('--output', '-o', default='requirements.txt', help='Output requirements file')
    p.add_argument('--from-file', dest='from_file', help='Read pip freeze output from this file instead of running pip freeze')
    p.add_argument('--exclude', '-e', action='append', help='Regex to exclude (can be used multiple times)')
    p.add_argument('--append-torch', action='store_true', help='Append placeholder install instruction for torch wheel')
    p.add_argument('--torch-instruction', default=None, help='Custom torch install instruction to append')
    return p.parse_args()


def get_frozen(from_file: str | None = None) -> List[str]:
    if from_file:
        p = Path(from_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {from_file}")
        return [line.rstrip('\n') for line in p.read_text(encoding='utf-8').splitlines()]
    # run pip freeze
    proc = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pip freeze failed: {proc.stderr}")
    return [line.rstrip('\n') for line in proc.stdout.splitlines()]


def filter_lines(lines: Iterable[str], exclude_patterns: Iterable[str]) -> List[str]:
    compiled = [re.compile(p, re.I) for p in exclude_patterns]
    out = []
    for line in lines:
        if not line or line.startswith('#'):
            continue
        name = line.split('==', 1)[0]
        if any(regex.search(name) for regex in compiled):
            continue
        out.append(line)
    return out


def main():
    args = parse_args()
    excludes = DEFAULT_EXCLUDES.copy()
    if args.exclude:
        excludes = args.exclude + excludes

    frozen = get_frozen(args.from_file)
    kept = filter_lines(frozen, excludes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for l in kept:
            f.write(l + '\n')
        if args.append_torch:
            note = args.torch_instruction
            if not note:
                note = (
                    '# Install torch separately using the appropriate CUDA wheel for your setup.\n'
                    "# Example (replace with the correct wheel URL and version):\n"
                    "# pip install https://download.pytorch.org/whl/cu128/torch-<version>+cu128-...whl\n"
                )
            f.write('\n')
            f.write(note + '\n')

    print(f"Wrote {len(kept)} packages to {out_path}")


if __name__ == '__main__':
    main()
