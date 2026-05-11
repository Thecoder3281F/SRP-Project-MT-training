import pandas as pd
import numpy as np
import json
import argparse
import logging
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

# Suppress RDKit warnings and logs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # pyright: ignore[reportAttributeAccessIssue]

def canonicalize(smiles):
    """Join tokens, parse to molecule, return canonical SMILES or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def tanimoto(a, b):
    """Compute Tanimoto similarity between two SMILES."""
    try:
        ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if not ma or not mb:
            return 0
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2) # pyright: ignore[reportAttributeAccessIssue]
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2) # pyright: ignore[reportAttributeAccessIssue]
        return DataStructs.TanimotoSimilarity(fa, fb)
    except Exception:
        return 0


def compute_metrics_from_csv(csv_path, gt_column="label", pred_columns=None, skip_tanimoto: bool = False):
    """
    Compute metrics from a CSV file with ground truth and predictions.

    Adds counts for invalid labels/predictions and returns both "all" and "valid-only" metrics.
    """

    if pred_columns is None:
        pred_columns = ["prediction_1", "prediction_2", "prediction_3", "prediction_4", "prediction_5"]

    # Load CSV
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Verify columns exist
    if gt_column not in df.columns:
        raise ValueError(f"Ground truth column '{gt_column}' not found in CSV")

    missing_cols = [col for col in pred_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction columns not found: {missing_cols}")

    # Metrics (all / baseline behavior)
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    top5_correct = 0
    tanimotos = []  # max tanimoto across predictions per row (zeros if no valid preds)
    valid_smiles_list = []  # flattened list of validity flags for all predictions
    # Track per-prediction tanimoto values (all rows)
    tanimoto_by_pred = {col: [] for col in pred_columns}

    # Counters for invalids and valid-only lists
    invalid_label_count = 0
    pred_invalid_counts = {col: 0 for col in pred_columns}
    rows_with_any_invalid_pred = 0
    rows_with_all_invalid_pred = 0
    num_samples = len(df)
    num_predictions = len(pred_columns)

    # For valid-only metrics
    num_label_valid_rows = 0
    tanimoto_by_pred_validonly = {col: [] for col in pred_columns}
    tanimotos_validonly = []  # per-row max over only valid predictions when label valid

    # Progress bar over rows
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics", unit="row"):
        label = row[gt_column]
        label_c = canonicalize(label)
        label_valid = bool(label_c)

        if not label_valid:
            invalid_label_count += 1
        else:
            num_label_valid_rows += 1

        found_top1 = found_top2 = found_top3 = found_top5 = False
        max_tani = 0
        max_tani_valid = None

        row_pred_valid_flags = []

        for rank, pred_col in enumerate(pred_columns):
            pred = row[pred_col]
            pred_c = canonicalize(pred)
            pred_valid = bool(pred_c)

            # Track invalid prediction counts
            if not pred_valid:
                pred_invalid_counts[pred_col] += 1

            row_pred_valid_flags.append(pred_valid)

            # Append a conservative validity flag (checks Molecule parse) for global validity fraction
            is_valid = int(Chem.MolFromSmiles(pred_c) is not None) if pred_c else 0
            valid_smiles_list.append(is_valid)

            # Compute tanimoto for this prediction vs ground truth (skip if requested)
            if not skip_tanimoto:
                tani = tanimoto(pred_c, label_c)
                tanimoto_by_pred[pred_col].append(tani)

                # Track tanimotos for valid-only (both label and pred must be valid)
                if label_valid and pred_valid:
                    tanimoto_by_pred_validonly[pred_col].append(tani)
                    if max_tani_valid is None or tani > max_tani_valid:
                        max_tani_valid = tani
            else:
                tani = 0

            # Top-k correctness (existing behaviour requires both canonical strings present and equal)
            if label_c and pred_c and pred_c == label_c:
                if rank < 1:
                    found_top1 = True
                if rank < 2:
                    found_top2 = True
                if rank < 3:
                    found_top3 = True
                if rank < 5:
                    found_top5 = True

            # Track max Tanimoto across the predictions for this row (all rows)
            if tani > max_tani:
                max_tani = tani

        # Row-level invalid stats
        if any(not f for f in row_pred_valid_flags):
            rows_with_any_invalid_pred += 1
        if all(not f for f in row_pred_valid_flags):
            rows_with_all_invalid_pred += 1

        # Aggregate
        top1_correct += int(found_top1)
        top2_correct += int(found_top2)
        top3_correct += int(found_top3)
        top5_correct += int(found_top5)
        tanimotos.append(max_tani)

        # Add valid-only per-row max if label valid and at least one pred valid
        if label_valid and (max_tani_valid is not None):
            tanimotos_validonly.append(max_tani_valid)

    # Calculate top-5 validity as: total valid SMILES / (num_samples * num_predictions)
    top5_validity_fraction = np.sum(valid_smiles_list) / (num_samples * num_predictions)
    top5_invalid_fraction = 1.0 - top5_validity_fraction

    # Compute per-prediction mean tanimoto (all rows)
    per_pred_tanimoto_means_all = {
        f"mean_tanimoto_{col}": float(np.mean(tanimoto_by_pred[col])) if len(tanimoto_by_pred[col]) > 0 else 0.0
        for col in pred_columns
    }

    # Compute per-prediction mean tanimoto (valid-only)
    per_pred_tanimoto_means_valid = {
        f"mean_tanimoto_valid_only_{col}": (float(np.mean(tanimoto_by_pred_validonly[col])) if len(tanimoto_by_pred_validonly[col]) > 0 else None)
        for col in pred_columns
    }

    # Overall means
    if not skip_tanimoto:
        mean_tanimoto_all = float(np.mean(tanimotos)) if len(tanimotos) > 0 else 0.0
        mean_tanimoto_valid_only = float(np.mean(tanimotos_validonly)) if len(tanimotos_validonly) > 0 else None
    else:
        mean_tanimoto_all = None
        mean_tanimoto_valid_only = None

    # Top-k: report both over all samples and valid-only (label valid only)
    canonical_top1_all = top1_correct / num_samples if num_samples > 0 else 0.0
    canonical_top2_all = top2_correct / num_samples if num_samples > 0 else 0.0
    canonical_top3_all = top3_correct / num_samples if num_samples > 0 else 0.0
    canonical_top5_all = top5_correct / num_samples if num_samples > 0 else 0.0

    canonical_top1_valid = top1_correct / num_label_valid_rows if num_label_valid_rows > 0 else None
    canonical_top2_valid = top2_correct / num_label_valid_rows if num_label_valid_rows > 0 else None
    canonical_top3_valid = top3_correct / num_label_valid_rows if num_label_valid_rows > 0 else None
    canonical_top5_valid = top5_correct / num_label_valid_rows if num_label_valid_rows > 0 else None

    # Build per-column counts and means (including-invalid and valid-only)
    per_column_counts = {}
    per_column_means_including_invalid = {}
    per_column_means_valid_only = {}

    for col in pred_columns:
        total = num_samples
        invalid = int(pred_invalid_counts.get(col, 0))
        valid = total - invalid
        per_column_counts[col] = {"total": total, "valid": valid, "invalid": invalid}

        # including-invalids mean is the mean across all rows (tanimoto_by_pred stored zeros for invalids)
        vals_all = tanimoto_by_pred[col]
        mean_incl = float(np.mean(vals_all)) if len(vals_all) > 0 else None
        per_column_means_including_invalid[col] = mean_incl

        # valid-only mean uses only entries where both label and pred were valid
        vals_valid = tanimoto_by_pred_validonly[col]
        mean_valid = float(np.mean(vals_valid)) if len(vals_valid) > 0 else None
        per_column_means_valid_only[col] = mean_valid

    counts = {
        "num_samples": num_samples,
        "num_label_valid": num_label_valid_rows,
        "invalid_label_count": invalid_label_count,
        "rows_with_any_invalid_prediction": rows_with_any_invalid_pred,
        "rows_with_all_invalid_predictions": rows_with_all_invalid_pred,
        "per_prediction_column": per_column_counts,
    }

    if not skip_tanimoto:
        tanimoto_section = {
            "overall": {
                "mean_including_invalid": mean_tanimoto_all,
                "mean_valid_only": mean_tanimoto_valid_only,
            },
            "per_column": {
                "including_invalid": per_column_means_including_invalid,
                "valid_only": per_column_means_valid_only,
            },
        }
    else:
        tanimoto_section = {}

    topk_section = {
        "all": {
            "canonical_top1": canonical_top1_all,
            "canonical_top2": canonical_top2_all,
            "canonical_top3": canonical_top3_all,
            "canonical_top5": canonical_top5_all,
        },
        "valid_only_label": {
            "canonical_top1": canonical_top1_valid,
            "canonical_top2": canonical_top2_valid,
            "canonical_top3": canonical_top3_valid,
            "canonical_top5": canonical_top5_valid,
        },
    }

    results = {
        "counts": counts,
        "topk": topk_section,
        "tanimoto": tanimoto_section,
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions from a CSV file")
    
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with ground truth and predictions")
    parser.add_argument("--gt_column", type=str, default="ground_truth",
                        help="Name of ground truth column (default: label)")
    parser.add_argument("--pred_columns", type=str, default="prediction_1,prediction_2,prediction_3,prediction_4,prediction_5",
                        help="Comma-separated list of prediction columns (default: prediction_1,prediction_2,prediction_3,prediction_4,prediction_5)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional output JSON file for results")
    parser.add_argument("--skip-tanimoto", action="store_true",
                        help="Skip Tanimoto similarity calculations (faster, omits tanimoto section)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse prediction columns
    pred_columns = [col.strip() for col in args.pred_columns.split(",")]
    
    # Compute metrics
    logger.info("Computing metrics...")
    results = compute_metrics_from_csv(args.csv_path, args.gt_column, pred_columns, skip_tanimoto=args.skip_tanimoto)
    
    # Print results as pretty JSON (handles nested structures safely)
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))
    print("="*60)
    
    # Save results if output file is specified (pretty + stable ordering)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
