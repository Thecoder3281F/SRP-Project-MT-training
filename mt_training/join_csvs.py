import pandas as pd
import argparse
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

# Suppress RDKit warnings and logs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    if pd.isna(smiles) or smiles == "":
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except Exception:
        return False


def validate_csv(df, csv_path, gt_column="ground_truth", pred_columns=None):
    """
    Validate CSV structure and SMILES validity.
    
    Args:
        df: DataFrame to validate
        csv_path: Path to CSV file
        gt_column: Name of ground truth column
        pred_columns: List of prediction column names
    
    Returns:
        Tuple of (is_valid, num_rows, num_invalid_smiles, invalid_smiles_count_by_col)
    """
    
    if pred_columns is None:
        pred_columns = ["pred1", "pred2", "pred3", "pred4", "pred5"]
    
    # Check required columns
    required_cols = [gt_column] + pred_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, 0, 0, {}
    
    # Check for invalid SMILES
    invalid_count_by_col = {}
    total_invalid = 0
    
    # Check ground truth
    invalid_gt = df[gt_column].apply(lambda x: not is_valid_smiles(x)).sum()
    if invalid_gt > 0:
        invalid_count_by_col[gt_column] = invalid_gt
        total_invalid += invalid_gt
    
    # Check predictions
    for col in pred_columns:
        invalid_pred = df[col].apply(lambda x: not is_valid_smiles(x)).sum()
        if invalid_pred > 0:
            invalid_count_by_col[col] = invalid_pred
            total_invalid += invalid_pred
    
    return True, len(df), total_invalid, invalid_count_by_col

def join_csvs(folder_path, output_path, gt_column="ground_truth", 
              pred_columns=None, skip_invalid=False):
    """
    Join all CSV files in a folder into one big CSV.
    
    Args:
        folder_path: Path to folder containing CSV files
        output_path: Path to output CSV file
        gt_column: Name of ground truth column
        pred_columns: List of prediction column names
        skip_invalid: If True, skip files with invalid SMILES
        validity_json_path: Optional path to save validity report as JSON
    """
    
    if pred_columns is None:
        pred_columns = ["prediction_1", "prediction_2", "prediction_3", "prediction_4", "prediction_5"]
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Find all CSV files
    csv_files = sorted(folder_path.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {folder_path}")
        return
    
    # Load and validate all CSVs
    dfs_to_join = []
    total_rows = 0
    total_invalid = 0
    
    # Track simple counters (no JSON report)
    files_processed = 0
    files_skipped = 0
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs", unit="file"):
        
        try:
            df = pd.read_csv(csv_file)
            is_valid, num_rows, num_invalid, invalid_by_col = validate_csv(
                df, csv_file, gt_column, pred_columns
            )
            
            if not is_valid:
                files_skipped += 1
                continue
            
            if num_invalid > 0:
                if skip_invalid:
                    files_skipped += 1
                    continue
            
            dfs_to_join.append(df)
            total_rows += num_rows
            total_invalid += num_invalid
            files_processed += 1
        
        except Exception as e:
            files_skipped += 1
            continue
    
    if not dfs_to_join:
        print("Error: No valid CSV files to join!")
        return
    
    # Join all DataFrames
    combined_df = pd.concat(dfs_to_join, ignore_index=True)
    
    # Save combined CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Combined CSV saved to: {output_path}")
    print(f"  Total rows: {combined_df.shape[0]}")
    print(f"  Total columns: {combined_df.shape[1]}")
    
    # Final summary (no JSON)
    if total_invalid > 0:
        print(f"\n⚠ Total invalid SMILES across all files: {total_invalid}")
    print(f"Processed files: {files_processed} | Skipped files: {files_skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join all CSV files in a folder into one big CSV file"
    )
    
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to folder containing CSV files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output combined CSV file")
    parser.add_argument("--gt_column", type=str, default="label",
                        help="Name of ground truth column (default: label)")
    parser.add_argument("--pred_columns", type=str, default="prediction_1,prediction_2,prediction_3,prediction_4,prediction_5",
                        help="Comma-separated list of prediction columns")
    parser.add_argument("--skip_invalid", action="store_true",
                        help="Skip files with invalid SMILES")
    # No JSON output option
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse prediction columns
    pred_columns = [col.strip() for col in args.pred_columns.split(",")]
    
    # Join CSVs
    join_csvs(
        args.folder,
        args.output,
        args.gt_column,
        pred_columns,
        args.skip_invalid
    )


if __name__ == "__main__":
    main()
