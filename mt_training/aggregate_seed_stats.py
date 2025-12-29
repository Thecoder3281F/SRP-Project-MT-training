"""
Aggregate per-seed summary CSVs (like `results_summary_unrounded.csv`) and compute
mean, sample SD (ddof=1), SEM, and 95% t-based CI per metric.

Usage examples:
  # Run on an existing set of per-seed CSVs under a root
  python mt_training/aggregate_seed_stats.py --root mt_training --pattern results_summary_unrounded.csv --out mt_training/aggregate_seed_stats.csv

  # Create dummy seed CSVs and run the aggregation (test mode)
  python mt_training/aggregate_seed_stats.py --create-dummy 3 --out mt_training/aggregate_seed_stats_test.csv

The script groups rows by `group_by` columns (default: `Full Model Name`) to aggregate
metrics across seeds that correspond to the same model.

Output: CSV with columns like `<metric>_mean`, `<metric>_sd`, `<metric>_sem`,
`<metric>_ci_lower`, `<metric>_ci_upper`, and `<metric>_n` for each numeric metric.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


def sample_sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float(np.nan)


def sem(x: np.ndarray) -> float:
    sd = sample_sd(x)
    return float(sd / math.sqrt(len(x))) if len(x) > 0 else float(np.nan)


def t_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(x)
    if n == 0:
        return (float('nan'), float('nan'))
    mean = float(np.mean(x))
    if n == 1:
        return (mean, mean)
    sd = sample_sd(x)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    return (float(mean - tcrit * se), float(mean + tcrit * se))


def infer_input_files(root: Path, pattern: str) -> List[Path]:
    files = sorted(root.rglob(pattern))
    return files


def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    # Select columns that are numeric in the DataFrame
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric


def aggregate(paths: List[Path], group_by: List[str], out: Path, sigfig: Optional[int]):
    if not paths:
        print("No files found to aggregate")
        return

    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"Skipping {p}: failed to read CSV: {e}")
            continue
        # add origin file info
        d['_seed_file'] = str(p)
        # if there is a chunk column with '# Samples' row we keep
        dfs.append(d)

    if not dfs:
        print("No valid CSV data loaded")
        return

    big = pd.concat(dfs, ignore_index=True, sort=False)

    # Grouping key exists in rows — default uses Full Model Name
    # If group_by columns aren't present, fall back to 'Full Model Name' or index
    for col in group_by:
        if col not in big.columns:
            print(f"Warning: group-by column '{col}' not found in CSVs; ignoring it")
    grp_cols = [c for c in group_by if c in big.columns]
    if not grp_cols:
        # fallback to Full Model Name if present
        if 'Full Model Name' in big.columns:
            grp_cols = ['Full Model Name']
        else:
            # fallback to first non-numeric column
            nonnum = [c for c in big.columns if big[c].dtype == object]
            if nonnum:
                grp_cols = [nonnum[0]]
            else:
                raise SystemExit('Cannot determine grouping columns')

    numeric_cols = find_numeric_columns(big)
    # exclude internal seed file column
    numeric_cols = [c for c in numeric_cols if c != '# Samples']

    agg_rows: List[Dict[str, Any]] = []

    grouped = big.groupby(grp_cols)
    for name, group in grouped:
        # name may be a scalar or tuple depending on number of group cols
        meta: Dict[str, Any] = {}
        if isinstance(name, tuple):
            for col, val in zip(grp_cols, name):
                meta[col] = val
        else:
            meta[grp_cols[0]] = name

        # For each numeric column compute stats across rows (these rows are per-seed rows)
        stats_out: Dict[str, Any] = {}
        for col in numeric_cols:
            arr = group[col].dropna().astype(float).values
            if len(arr) == 0:
                mean = float('nan')
                sd = float('nan')
                se = float('nan')
                ci_low = float('nan')
                ci_high = float('nan')
                n = 0
            else:
                mean = float(np.mean(arr))
                sd = sample_sd(arr) if len(arr) > 1 else 0.0
                se = sem(arr) if len(arr) > 0 else float('nan')
                ci_low, ci_high = t_ci(arr)
                n = len(arr)
            # rounding if requested
            if sigfig is not None and not math.isnan(mean):
                # use round-significant by converting to string? We'll use numpy formatting for clarity
                # keep numeric types but reduce digits via python round on significant
                def round_sig(x, s):
                    if x == 0 or not math.isfinite(x):
                        return x
                    mag = math.floor(math.log10(abs(x)))
                    nd = s - mag - 1
                    return round(x, nd)
                mean = round_sig(mean, sigfig)
                sd = round_sig(sd, sigfig) if not math.isnan(sd) else sd
                se = round_sig(se, sigfig) if not math.isnan(se) else se
                ci_low = round_sig(ci_low, sigfig) if not math.isnan(ci_low) else ci_low
                ci_high = round_sig(ci_high, sigfig) if not math.isnan(ci_high) else ci_high
            stats_out[f"{col}_mean"] = mean
            stats_out[f"{col}_sd"] = sd
            stats_out[f"{col}_sem"] = se
            stats_out[f"{col}_ci_lower"] = ci_low
            stats_out[f"{col}_ci_upper"] = ci_high
            stats_out[f"{col}_n"] = n

        row = {**meta, **stats_out}
        agg_rows.append(row)

    out_df = pd.DataFrame(agg_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} aggregated rows to {out}")


def create_dummy_seed_csvs(root: Path, n_seeds: int = 3):
    # create a simple structure: root/seed_<i>/results_summary_unrounded.csv
    models = [
        {"Full Model Name": "MTC-T5 small", "Family": "MTC-T5", "Size": "Small", "Augmented?": "No"},
        {"Full Model Name": "Manganum-base-v1.1-standard", "Family": "Manganum-v1.1", "Size": "Base", "Augmented?": "No"},
    ]
    for i in range(n_seeds):
        seed_dir = root / f"seed_{i+1}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for m in models:
            # vary numbers slightly per seed
            top1 = 0.7 + 0.05 * np.random.randn()
            mean_tani = 0.85 + 0.02 * np.random.randn()
            row = {
                **m,
                "# Samples": 960,
                "Rows with any invalid": int(max(0, 200 + np.random.randint(-50, 50))),
                "Rows with all invalid": 0,
                "Invalid % (any rank)": float((np.random.rand() * 10.0)),
                "Top-1 Accuracy": float(top1 * 100.0),
                "Top-2 Accuracy": float(min(100.0, (top1 + 0.1) * 100.0)),
                "Top-3 Accuracy": float(min(100.0, (top1 + 0.15) * 100.0)),
                "Top-5 Accuracy": float(min(100.0, (top1 + 0.2) * 100.0)),
                "Mean Tanimoto (valid)": float(mean_tani),
                "Mean Tanimoto (incl. invalid)": float(mean_tani * 0.99),
                "Prediction 1 Invalid": int(np.random.randint(0, 50)),
                "Prediction 1 Tanimoto (valid)": float(mean_tani + np.random.randn() * 0.01),
                "Prediction 1 Tanimoto (incl invalid)": float(mean_tani + np.random.randn() * 0.01),
                "Prediction 2 Invalid": int(np.random.randint(0, 200)),
                "Prediction 2 Tanimoto (valid)": float(mean_tani - 0.2 + np.random.randn() * 0.02),
                "Prediction 2 Tanimoto (incl invalid)": float(mean_tani - 0.2 + np.random.randn() * 0.02),
                "Prediction 3 Invalid": int(np.random.randint(0, 250)),
                "Prediction 3 Tanimoto (valid)": float(mean_tani - 0.25 + np.random.randn() * 0.02),
                "Prediction 3 Tanimoto (incl invalid)": float(mean_tani - 0.25 + np.random.randn() * 0.02),
                "Prediction 4 Invalid": int(np.random.randint(0, 300)),
                "Prediction 4 Tanimoto (valid)": float(mean_tani - 0.3 + np.random.randn() * 0.02),
                "Prediction 4 Tanimoto (incl invalid)": float(mean_tani - 0.3 + np.random.randn() * 0.02),
                "Prediction 5 Invalid": int(np.random.randint(0, 350)),
                "Prediction 5 Tanimoto (valid)": float(mean_tani - 0.35 + np.random.randn() * 0.02),
                "Prediction 5 Tanimoto (incl invalid)": float(mean_tani - 0.35 + np.random.randn() * 0.02),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(seed_dir / 'results_summary_unrounded.csv', index=False)
    print(f"Created {n_seeds} dummy seed CSVs under {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='mt_training', help='Root to search for per-seed CSVs')
    parser.add_argument('--pattern', type=str, default='results_summary_unrounded.csv', help='Filename pattern to search')
    parser.add_argument('--out', type=str, default='mt_training/aggregate_seed_stats.csv', help='Output aggregated CSV path')
    parser.add_argument('--group-by', type=str, default='Full Model Name', help='Comma-separated group-by columns')
    parser.add_argument('--sigfig', type=int, default=0, help='Round output to N significant figures (0 = no rounding)')
    parser.add_argument('--create-dummy', type=int, default=0, help='Create dummy seed CSVs (count) under <root>/dummy_seeds and run aggregation')
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    group_by = [x.strip() for x in args.group_by.split(',') if x.strip()]
    sigfig = None if args.sigfig == 0 else int(args.sigfig)

    if args.create_dummy and args.create_dummy > 0:
        dummy_root = root / 'dummy_seeds'
        create_dummy_seed_csvs(dummy_root, args.create_dummy)
        files = sorted(dummy_root.rglob(args.pattern))
        aggregate(files, group_by, out, sigfig)
    else:
        files = infer_input_files(root, args.pattern)
        aggregate(files, group_by, out, sigfig)


if __name__ == '__main__':
    main()
