"""Configurable runner for progressive subset generation.

This script runs `make_progressive_group_subsets.py` multiple times with
one or more seeds, writing each run to its own output directory.

Typical use cases:
- Generate the same split sizes for multiple subset-generation seeds
- Compare subset construction variance separately from training variance
- Keep all generated subsets in a predictable folder layout

Examples:

1) One seed, row-based sizes, spread sampling:

    python mt_training/run_progressive_subset_grid.py \
        --input datasets_final/chanlam_final/train_final.csv \
        --out-root datasets_final/chanlam_final/progressive_seeds \
        --subset-seeds 42 \
        --sizes-rows 10,50,100,250,500,1000 \
        --sample-mode spread \
        --diversity-col product_1_canonical_smiles \
        --per-group 15

2) Three subset seeds, auto per-group sizing:

    python mt_training/run_progressive_subset_grid.py \
        --input datasets_final/chanlam_final/train_final.csv \
        --out-root datasets_final/chanlam_final/progressive_seeds \
        --subset-seeds 42,43,44 \
        --sizes-rows 10,50,100,250,500,1000,2500,5000 \
        --sample-mode spread \
        --auto-per-group

Each seed will be written to a subfolder like:
    <out-root>/seed_42/
    <out-root>/seed_43/
    ...
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

HERE = Path(__file__).resolve().parent
SUBSET_SCRIPT = HERE / "make_progressive_group_subsets.py"
PYTHON = sys.executable


def parse_csv_ints(value: str) -> List[int]:
    if not value or not value.strip():
        return []
    out: List[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise SystemExit(f"Invalid integer '{tok}'") from exc
    if not out:
        raise SystemExit("No valid integers parsed")
    return out


def parse_csv_strs(value: str) -> List[str]:
    if not value or not value.strip():
        return []
    out = [tok.strip() for tok in value.split(",") if tok.strip()]
    if not out:
        raise SystemExit("No valid values parsed")
    return out


def build_base_command(args, seed: int, out_dir: Path) -> List[str]:
    cmd: List[str] = [
        PYTHON,
        str(SUBSET_SCRIPT),
        "--input",
        args.input,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--sample-mode",
        args.sample_mode,
    ]

    if args.group_col:
        cmd.extend(["--group-col", args.group_col])
    if args.sizes:
        cmd.extend(["--sizes", args.sizes])
    if args.sizes_rows:
        cmd.extend(["--sizes-rows", args.sizes_rows])
    if args.diversity_col:
        cmd.extend(["--diversity-col", args.diversity_col])
    if args.per_group is not None:
        cmd.extend(["--per-group", str(args.per_group)])
    if args.auto_per_group:
        cmd.append("--auto-per-group")
    if args.no_canonicalise:
        cmd.append("--no-canonicalise")
    if args.exclude_full:
        cmd.append("--exclude-full")

    return cmd


def print_command(cmd: Iterable[str]) -> None:
    if os.name == "nt":
        # Windows-friendly shell display
        print(" ".join(shlex.quote(part) for part in cmd))
    else:
        print(" ".join(shlex.quote(part) for part in cmd))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run make_progressive_group_subsets.py across multiple subset-generation seeds."
    )
    parser.add_argument("--input", required=True, help="Input training CSV")
    parser.add_argument("--out-root", required=True, help="Root directory for seed subfolders")
    parser.add_argument(
        "--subset-seeds",
        required=True,
        help="Comma-separated list of subset-generation seeds (e.g. 42,43,44)",
    )
    parser.add_argument(
        "--sizes",
        default="",
        help="Comma-separated target subset sizes in groups (passed through to make_progressive_group_subsets.py)",
    )
    parser.add_argument(
        "--sizes-rows",
        default="",
        help="Comma-separated target subset sizes in rows (passed through to make_progressive_group_subsets.py)",
    )
    parser.add_argument(
        "--sample-mode",
        default="spread",
        choices=["representative", "all-group-rows", "diverse", "spread"],
        help="Subset selection mode to pass through to the subset generator",
    )
    parser.add_argument("--group-col", default=None, help="Group column to pass through")
    parser.add_argument(
        "--diversity-col",
        default="product_1_canonical_smiles",
        help="SMILES column used by diversity-aware modes",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=1,
        help="Rows per group when using diverse/spread modes",
    )
    parser.add_argument(
        "--auto-per-group",
        action="store_true",
        help="Let the subset generator compute the minimal per-group count per target row size",
    )
    parser.add_argument(
        "--no-canonicalise",
        action="store_true",
        help="Forward --no-canonicalise to the subset generator",
    )
    parser.add_argument(
        "--exclude-full",
        action="store_true",
        help="Forward --exclude-full to the subset generator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing seed output folders if they already exist",
    )

    args = parser.parse_args()
    seed_list = parse_csv_ints(args.subset_seeds)
    if not seed_list:
        raise SystemExit("--subset-seeds must contain at least one integer")

    if not Path(args.input).exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for seed in seed_list:
        seed_dir = Path(args.out_root) / f"seed_{seed}"
        if seed_dir.exists():
            if args.overwrite:
                # remove old contents only within the seed folder
                for child in seed_dir.iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        import shutil

                        shutil.rmtree(child)
            else:
                raise SystemExit(
                    f"Output directory already exists: {seed_dir}. Use --overwrite to reuse it."
                )
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_base_command(args, seed=seed, out_dir=seed_dir)
        print(f"\n=== Seed {seed} ===")
        print_command(cmd)

        if args.dry_run:
            status = "dry-run"
            rc = 0
            stdout = ""
            stderr = ""
        else:
            proc = subprocess.run(cmd, text=True, capture_output=True)
            rc = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
            status = "ok" if rc == 0 else "failed"
            print(f"Return code: {rc}")
            if stdout.strip():
                print("--- STDOUT ---")
                print(stdout)
            if stderr.strip():
                print("--- STDERR ---")
                print(stderr)

        summary_rows.append(
            {
                "seed": seed,
                "out_dir": str(seed_dir),
                "status": status,
                "returncode": rc,
            }
        )

        if rc != 0 and not args.dry_run:
            raise SystemExit(rc)

    summary_path = Path(args.out_root) / "runner_summary.csv"
    try:
        import pandas as pd

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nWrote summary: {summary_path}")
    except Exception as exc:
        print(f"Warning: could not write summary CSV: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
