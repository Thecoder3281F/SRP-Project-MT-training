#!/usr/bin/env python3
"""Streamlit app to interactively view subset UMAPs with checkbox menu.

Run with:
    streamlit run visualisation_scripts/streamlit_subsets.py

Requirements: streamlit, rdkit, umap-learn, plotly, pandas, numpy
"""
import os
import glob
from typing import List

import streamlit as st
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except Exception:
    Chem = None
    AllChem = None
    DataStructs = None

try:
    import umap
except Exception:
    umap = None

import plotly.express as px


def find_subset_files(dirpath: str) -> List[str]:
    files = sorted(
        glob.glob(os.path.join(dirpath, "train_rows_*.csv")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
    )
    return files


@st.cache_data(show_spinner=False)
def compute_fps(smiles: List[str], n_bits: int = 2048, radius: int = 2):
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    if Chem is None or AllChem is None or DataStructs is None:
        return X
    for i, s in enumerate(smiles):
        try:
            if pd.isna(s) or not str(s).strip():
                continue
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            vec = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, vec)
            X[i, :] = vec
        except Exception:
            continue
    return X


@st.cache_data(show_spinner=False)
def run_umap(X: np.ndarray, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    if umap is None:
        raise RuntimeError("umap-learn not available")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric="jaccard", random_state=random_state)
    embedding = reducer.fit_transform(X)
    return embedding


def load_or_compute_coords(dirpath: str, smiles_col: str, max_per_file: int, random_state: int):
    files = find_subset_files(dirpath)
    if not files:
        st.error(f"No subset files found in {dirpath}")
        return None

    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        if max_per_file and len(df) > max_per_file:
            df = df.sample(n=max_per_file, random_state=random_state)
        df = df.reset_index(drop=True)
        df['_subset_file'] = os.path.basename(f)
        df['_subset_order'] = i
        frames.append(df)

    bigdf = pd.concat(frames, ignore_index=True)
    smiles = bigdf[smiles_col].fillna("").tolist()

    coords_file = os.path.join(dirpath, "umap_coords_streamlit.csv")
    if os.path.exists(coords_file):
        try:
            coords = pd.read_csv(coords_file)
            return coords
        except Exception:
            pass

    if Chem is None or umap is None:
        st.warning("RDKit and umap-learn required to compute coordinates. Install them or provide precomputed coords.")
        return None

    with st.spinner("Computing fingerprints..."):
        X = compute_fps(smiles)
    with st.spinner("Running UMAP (may take a moment)..."):
        embedding = run_umap(X, random_state=random_state)

    coords = bigdf.loc[X.sum(axis=1) > 0].reset_index(drop=True).copy()
    coords['umap_x'] = embedding[:, 0]
    coords['umap_y'] = embedding[:, 1]
    coords.to_csv(coords_file, index=False)
    return coords


def main():
    st.set_page_config(page_title="Subset UMAP Explorer", layout="wide")
    st.title("Subset UMAP Explorer")

    with st.sidebar:
        st.header("Data")
        dirpath = st.text_input("Subsets directory", value="datasets_final/chanlam_final/progressive_by_reactants")
        smiles_col = st.text_input("SMILES column", value="product_1_canonical_smiles")
        max_per_file = st.number_input("Max rows per file (0 = all)", min_value=0, value=200)
        max_per_file = None if max_per_file == 0 else int(max_per_file)
        random_state = st.number_input("Random state", value=42)
        st.markdown("---")
        st.header("UMAP settings")
        n_neighbors = st.number_input("n_neighbors", value=15)
        min_dist = st.number_input("min_dist", value=0.1, format="%.2f")
        recompute = st.button("(Re)compute UMAP now")

    files = find_subset_files(dirpath)
    if not files:
        st.error("No subset files found — check the directory path")
        return

    selected = st.multiselect("Select subset files to show", options=[os.path.basename(p) for p in files], default=[os.path.basename(p) for p in files])

    coords = None
    if os.path.exists(os.path.join(dirpath, "umap_coords_streamlit.csv")) and not recompute:
        coords = pd.read_csv(os.path.join(dirpath, "umap_coords_streamlit.csv"))
    else:
        coords = load_or_compute_coords(dirpath, smiles_col, max_per_file, int(random_state))

    if coords is None:
        return

    # filter to selected
    coords = coords[coords['_subset_file'].isin(selected)]

    if coords.empty:
        st.warning("No points to display for selected subsets")
        return

    fig = px.scatter(
        coords,
        x='umap_x',
        y='umap_y',
        color='_subset_file',
        symbol='_subset_file',
        hover_data=[smiles_col, '_subset_file'],
        title='UMAP of selected subsets',
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Download visible coords CSV"):
        st.download_button("Download coords", coords.to_csv(index=False), file_name="visible_coords.csv")


if __name__ == '__main__':
    main()
