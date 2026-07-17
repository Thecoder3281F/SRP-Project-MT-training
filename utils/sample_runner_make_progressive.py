"""TODO: Add module docstring describing this helper script."""

import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "temp_sample_input.csv")
OUT_DIR = os.path.join(HERE, "temp_out_diverse")
PY = sys.executable

# Build a small toy dataframe with chemically-valid SMILES for reactants and products
rows = [
    {"input_reactants": "c1ccccc1", "input_reagents": "Br", "product_1_canonical_smiles": "c1ccccc1Br"},
    {"input_reactants": "c1ccccc1", "input_reagents": "O", "product_1_canonical_smiles": "Oc1ccccc1"},
    {"input_reactants": "CCO", "input_reagents": "Cl", "product_1_canonical_smiles": "CCCl"},
    {"input_reactants": "CCO", "input_reagents": "Br", "product_1_canonical_smiles": "CCBr"},
    {"input_reactants": "CC(=O)O", "input_reagents": "C", "product_1_canonical_smiles": "CC(=O)OC"},
    {"input_reactants": "CC(=O)O", "input_reagents": "O", "product_1_canonical_smiles": "CC(=O)O"},
    {"input_reactants": "NCCO", "input_reagents": "H", "product_1_canonical_smiles": "NCCO"},
]

df = pd.DataFrame(rows)
df.to_csv(INPUT_CSV, index=False)

os.makedirs(OUT_DIR, exist_ok=True)

cmd = [
    PY,
    os.path.join(HERE, "make_progressive_group_subsets.py"),
    "--input",
    INPUT_CSV,
    "--out-dir",
    OUT_DIR,
    "--sizes",
    "1,2,3",
    "--sample-mode",
    "diverse",
    "--diversity-col",
    "product_1_canonical_smiles",
    "--per-group",
    "1",
    "--seed",
    "42",
]

print("Running:", " ".join(cmd))
res = subprocess.run(cmd, text=True, capture_output=True)
print("RETURN_CODE=", res.returncode)
print("--- STDOUT ---")
print(res.stdout)
print("--- STDERR ---")
print(res.stderr)

print("Output files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", os.path.join(OUT_DIR, f))

# Verification: print selected rows per generated subset
print("\nVerification: selected rows per group in each subset")
for fn in sorted(os.listdir(OUT_DIR)):
    if not fn.startswith("train_groups_") or not fn.endswith(".csv"):
        continue
    path = os.path.join(OUT_DIR, fn)
    sdf = pd.read_csv(path)
    print(f"\nFile: {fn} (rows={len(sdf)})")
    if "input_reactants" in sdf.columns and "product_1_canonical_smiles" in sdf.columns:
        for g, sub in sdf.groupby("input_reactants"):
            prods = sub["product_1_canonical_smiles"].tolist()
            print(f" - group={g}: selected products={prods}")
    else:
        print(" - Unexpected columns in subset CSV")
