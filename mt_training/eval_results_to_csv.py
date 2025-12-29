"""mt_training/eval_results_to_csv.py

Build a summary CSV (like `SRP results - ChanLam actual results.csv`) from one or many
`evaluation_results.json` files produced by `eval_from_csv.py`.

Default behavior scans all `evaluation_results.json` under `mt_training/model_predictions`.

Example:
    python .\\mt_training\\eval_results_to_csv.py --root mt_training\\model_predictions --out mt_training\\results_summary.csv

Notes:
- Top-k accuracies are written as percentages (0-100).
- "Invalid % (any rank)" is computed as (sum invalid predictions across ranks) / (num_samples * 5) * 100.
- Family/Size/Augmented? are inferred from the folder name with simple heuristics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


COLUMNS = [
    "Family",
    "Size",
    "Augmented?",
    "Full Model Name",
    "# Samples",
    "Rows with any invalid",
    "Rows with all invalid",
    "Invalid % (any rank)",
    "Top-1 Accuracy",
    "Top-2 Accuracy",
    "Top-3 Accuracy",
    "Top-5 Accuracy",
    "Mean Tanimoto (valid)",
    "Mean Tanimoto (incl. invalid)",
    "Prediction 1 Invalid",
    "Prediction 1 Tanimoto (valid)",
    "Prediction 1 Tanimoto (incl invalid)",
    "Prediction 2 Invalid",
    "Prediction 2 Tanimoto (valid)",
    "Prediction 2 Tanimoto (incl invalid)",
    "Prediction 3 Invalid",
    "Prediction 3 Tanimoto (valid)",
    "Prediction 3 Tanimoto (incl invalid)",
    "Prediction 4 Invalid",
    "Prediction 4 Tanimoto (valid)",
    "Prediction 4 Tanimoto (incl invalid)",
    "Prediction 5 Invalid",
    "Prediction 5 Tanimoto (valid)",
    "Prediction 5 Tanimoto (incl invalid)",
]


def round_sig(x: float, sig: int = 3) -> float:
    if x == 0:
        return 0.0
    if not math.isfinite(x):
        return x
    magnitude = math.floor(math.log10(abs(x)))
    ndigits = sig - magnitude - 1
    return float(round(x, ndigits))


def maybe_round(v: Any, sig: Optional[int]) -> Any:
    if sig is None:
        return v
    if isinstance(v, bool) or isinstance(v, int) or v is None:
        return v
    if isinstance(v, float):
        return round_sig(v, sig)
    return v


def infer_metadata(eval_json_path: Path) -> Dict[str, str]:
    """Infer Family/Size/Augmented?/Full Model Name from folder names."""
    # evaluation_results.json typically lives at: .../<model>/<split>/evaluation_results.json
    split_folder = eval_json_path.parent
    model_folder = split_folder.parent

    model_name = model_folder.name
    split_name = split_folder.name

    model_lc = model_name.lower()

    # Size
    if "small" in model_lc:
        size = "Small"
    elif "base" in model_lc:
        size = "Base"
    else:
        size = ""  # unknown

    # Augmented?
    augmented = "Yes" if "aug" in model_lc or "augmented" in model_lc else "No"

    # Family (heuristics)
    if "multitasktextandchemistry" in model_lc or "mtc" in model_lc:
        family = "MTC-T5"
    elif "reactiont5" in model_lc:
        family = "ReactionT5"
    elif "manganum" in model_lc or "manganumt5" in model_lc:
        # try to detect v1.1 style
        family = "Manganum-v1.1" if ("v1_1" in model_lc or "v1.1" in model_lc) else "Manganum"
    else:
        # fallback: first token before '-' or '_'
        family = re.split(r"[-_]+", model_name)[0]

    # Full model name: keep it readable, include split if meaningful
    full_name = model_name
    if split_name and split_name not in {".", ""}:
        full_name = f"{model_name}/{split_name}"

    return {
        "Family": family,
        "Size": size,
        "Augmented?": augmented,
        "Full Model Name": full_name,
    }


def get_nested(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_row(eval_path: Path, data: Dict[str, Any], sigfig: Optional[int]) -> Dict[str, Any]:
    meta = infer_metadata(eval_path)

    num_samples = int(get_nested(data, "counts", "num_samples", default=0) or 0)
    rows_any_invalid = int(get_nested(data, "counts", "rows_with_any_invalid_prediction", default=0) or 0)
    rows_all_invalid = int(get_nested(data, "counts", "rows_with_all_invalid_predictions", default=0) or 0)

    # Invalid % (any rank) = total invalid predictions across ranks / (N * 5) * 100
    per_col = get_nested(data, "counts", "per_prediction_column", default={}) or {}
    invalid_sum = 0
    for i in range(1, 6):
        col = f"prediction_{i}"
        invalid_sum += int(get_nested(per_col, col, "invalid", default=0) or 0)
    invalid_pct_any_rank = (invalid_sum / float(num_samples * 5) * 100.0) if num_samples else 0.0

    # Top-k (stored as fraction)
    top1 = get_nested(data, "topk", "all", "canonical_top1", default=None)
    top2 = get_nested(data, "topk", "all", "canonical_top2", default=None)
    top3 = get_nested(data, "topk", "all", "canonical_top3", default=None)
    top5 = get_nested(data, "topk", "all", "canonical_top5", default=None)

    def pct(x):
        return (float(x) * 100.0) if x is not None else None

    # Mean tanimoto
    mean_valid = get_nested(data, "tanimoto", "overall", "mean_valid_only", default=None)
    mean_incl = get_nested(data, "tanimoto", "overall", "mean_including_invalid", default=None)

    row: Dict[str, Any] = {
        **meta,
        "# Samples": num_samples,
        "Rows with any invalid": rows_any_invalid,
        "Rows with all invalid": rows_all_invalid,
        "Invalid % (any rank)": maybe_round(invalid_pct_any_rank, sigfig),
        "Top-1 Accuracy": maybe_round(pct(top1), sigfig),
        "Top-2 Accuracy": maybe_round(pct(top2), sigfig),
        "Top-3 Accuracy": maybe_round(pct(top3), sigfig),
        "Top-5 Accuracy": maybe_round(pct(top5), sigfig),
        "Mean Tanimoto (valid)": maybe_round(float(mean_valid) if mean_valid is not None else None, sigfig),
        "Mean Tanimoto (incl. invalid)": maybe_round(float(mean_incl) if mean_incl is not None else None, sigfig),
    }

    per_valid = get_nested(data, "tanimoto", "per_column", "valid_only", default={}) or {}
    per_incl = get_nested(data, "tanimoto", "per_column", "including_invalid", default={}) or {}

    for i in range(1, 6):
        pred_col = f"prediction_{i}"
        invalid_count = int(get_nested(per_col, pred_col, "invalid", default=0) or 0)
        tani_valid = per_valid.get(pred_col, None)
        tani_incl = per_incl.get(pred_col, None)

        row[f"Prediction {i} Invalid"] = invalid_count
        row[f"Prediction {i} Tanimoto (valid)"] = maybe_round(float(tani_valid) if tani_valid is not None else None, sigfig)
        row[f"Prediction {i} Tanimoto (incl invalid)"] = maybe_round(float(tani_incl) if tani_incl is not None else None, sigfig)

    # Ensure all columns exist
    for c in COLUMNS:
        row.setdefault(c, None)

    return row


def main():
    parser = argparse.ArgumentParser(description="Convert evaluation_results.json files into a summary CSV")
    parser.add_argument("--root", type=str, default="mt_training/model_predictions", help="Root folder to search")
    parser.add_argument("--out", type=str, default="mt_training/results_summary.csv", help="Output CSV path")
    parser.add_argument("--sigfig", type=int, default=3, help="Round floats to N significant figures (default: 3). Use 0 to disable.")
    parser.add_argument("--pattern", type=str, default="evaluation_results.json", help="Filename to match (default: evaluation_results.json)")
    args = parser.parse_args()

    sigfig: Optional[int] = None if args.sigfig == 0 else int(args.sigfig)

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    paths = sorted(root.rglob(args.pattern))
    if not paths:
        raise SystemExit(f"No files named {args.pattern} found under {root}")

    rows = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {p}: failed to read JSON: {e}")
            continue
        try:
            rows.append(build_row(p, data, sigfig))
        except Exception as e:
            print(f"Skipping {p}: failed to parse metrics: {e}")

    df = pd.DataFrame(rows)
    # Order columns exactly like the manual CSV
    df = df[COLUMNS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
