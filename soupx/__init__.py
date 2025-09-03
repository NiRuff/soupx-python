"""
SoupX Python Implementation

A Python port of the SoupX R package for removing ambient RNA contamination
from droplet-based single-cell RNA sequencing data.

This implementation mirrors the R package behavior exactly.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from .core import SoupChannel
from .estimation import (
    autoEstCont,
    estimateNonExpressingCells,
    quickMarkers
)
from .correction import adjustCounts

__version__ = "0.3.0"

# R-compatible naming (primary interface)
__all__ = [
    "SoupChannel",
    "adjustCounts",
    "autoEstCont",
    "estimateNonExpressingCells",
    "quickMarkers",
]


# Convenience functions for Python users (snake_case aliases)
def adjust_counts(*args, **kwargs):
    """Python-style alias for adjustCounts."""
    return adjustCounts(*args, **kwargs)


def auto_est_cont(*args, **kwargs):
    """Python-style alias for autoEstCont."""
    return autoEstCont(*args, **kwargs)


def estimate_non_expressing_cells(*args, **kwargs):
    """Python-style alias for estimateNonExpressingCells."""
    return estimateNonExpressingCells(*args, **kwargs)


def quick_markers(*args, **kwargs):
    """Python-style alias for quickMarkers."""
    return quickMarkers(*args, **kwargs)


def load10X(dataDir, **kwargs):
    """
    Load 10X data from cellranger output directory.
    Mimics R's load10X function.

    Parameters
    ----------
    dataDir : str
        Path to cellranger outs folder
    **kwargs
        Additional arguments passed to SoupChannel

    Returns
    -------
    SoupChannel
        Initialized SoupChannel object
    """
    import scanpy as sc
    import os
    from pathlib import Path

    data_path = Path(dataDir)

    # Check for different cellranger output structures
    if (data_path / "filtered_feature_bc_matrix").exists():
        # cellranger v3+
        toc = sc.read_10x_mtx(data_path / "filtered_feature_bc_matrix")
        tod = sc.read_10x_mtx(data_path / "raw_feature_bc_matrix")
    elif (data_path / "filtered_gene_bc_matrices").exists():
        # cellranger v2
        # Find genome folder
        genome_dir = list((data_path / "filtered_gene_bc_matrices").iterdir())[0]
        toc = sc.read_10x_mtx(data_path / "filtered_gene_bc_matrices" / genome_dir.name)
        tod = sc.read_10x_mtx(data_path / "raw_gene_bc_matrices" / genome_dir.name)
    else:
        raise ValueError(f"Could not find 10X data in {dataDir}")

    # Convert to sparse CSR matrices
    toc_sparse = toc.X.T.tocsr()  # Transpose to genes x cells
    tod_sparse = tod.X.T.tocsr()

    # Create metadata
    metaData = pd.DataFrame({
        'nUMIs': np.array(toc_sparse.sum(axis=0)).flatten()
    }, index=toc.obs_names)

    # Try to load clusters if available
    clusters_path = data_path / "analysis" / "clustering" / "graphclust" / "clusters.csv"
    if clusters_path.exists():
        clusters_df = pd.read_csv(clusters_path)
        # Match barcodes
        if 'Barcode' in clusters_df.columns and 'Cluster' in clusters_df.columns:
            cluster_dict = dict(zip(clusters_df['Barcode'], clusters_df['Cluster']))
            metaData['clusters'] = [str(cluster_dict.get(bc, '0')) for bc in toc.obs_names]

    # Create SoupChannel
    sc_obj = SoupChannel(
        tod=tod_sparse,
        toc=toc_sparse,
        metaData=metaData,
        **kwargs
    )

    # Store gene names if available
    if hasattr(toc, 'var_names'):
        sc_obj.gene_names = toc.var_names.tolist()
        # Set soup profile index to gene names
        if sc_obj.soupProfile is not None:
            sc_obj.soupProfile.index = sc_obj.gene_names

    return sc_obj



def calculate_contamination_fraction(sc, non_expressed_genes, clusters=None):
    """Simple contamination estimation for backwards compatibility."""
    # Just call autoEstCont with defaults
    return autoEstCont(sc, verbose=False)