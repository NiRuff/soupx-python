"""
Soup profile estimation functions.

Core functionality for estimating the background contamination profile
from empty droplets.
"""
import numpy as np
import pandas as pd
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SoupChannel


def estimate_soup(
        sc: "SoupChannel",
        soup_range: Tuple[int, int] = (0, 100),
        keep_raw_counts: bool = False
) -> "SoupChannel":
    """
    Estimate soup profile from empty droplets.

    Implements equation (1) from SoupX paper:
    bg = sum_d(ng,d) / sum_g,d(ng,d)

    Uses droplets with UMI counts in soup_range as "empty" droplets
    to estimate the background expression profile.

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with raw and filtered counts
    soup_range : tuple of int, default (0, 100)
        Range of UMI counts to consider as empty droplets (exclusive bounds)
    keep_raw_counts : bool, default False
        Whether to keep raw counts matrix after estimation (saves memory if False)

    Returns
    -------
    SoupChannel
        Modified SoupChannel with soup_profile attribute set
    """

    # Find empty droplets based on UMI count range
    raw_umi_counts = sc.raw_umi_counts
    empty_droplet_mask = (raw_umi_counts > soup_range[0]) & (raw_umi_counts < soup_range[1])

    if not np.any(empty_droplet_mask):
        raise ValueError(f"No droplets found in soup_range {soup_range}")

    n_empty = np.sum(empty_droplet_mask)
    print(f"Using {n_empty} droplets with UMI counts in range {soup_range} to estimate soup")

    # Calculate soup profile from empty droplets
    empty_counts = sc.raw_counts[:, empty_droplet_mask]

    # Sum counts across all empty droplets for each gene
    gene_counts_in_soup = np.array(empty_counts.sum(axis=1)).flatten()
    total_counts_in_soup = np.sum(gene_counts_in_soup)

    if total_counts_in_soup == 0:
        raise ValueError("No counts found in empty droplets")

    # Calculate soup profile (fraction of each gene in the soup)
    soup_fractions = gene_counts_in_soup / total_counts_in_soup

    # Create soup profile DataFrame
    soup_profile = pd.DataFrame(
        index=sc.gene_names,
        data={
            'est': soup_fractions,  # Estimated soup fraction for each gene
            'counts': gene_counts_in_soup  # Raw counts in soup for each gene
        }
    )

    sc.soup_profile = soup_profile

    # Optionally drop raw counts to save memory
    if not keep_raw_counts:
        sc.raw_counts = None

    return sc


def calculate_contamination_fraction(
        sc: "SoupChannel",
        non_expressed_genes: list,
        non_expressing_cells: np.ndarray = None
) -> "SoupChannel":
    """
    Calculate contamination fraction using non-expressed genes.

    Implements equation (4) from SoupX paper:
    ρc = (sum_g ng,c) / (Nc * sum_g bg)

    For genes that should not be expressed in certain cells.

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with soup profile
    non_expressed_genes : list
        List of gene names that should not be expressed in some cells
    non_expressing_cells : array of bool, optional
        Boolean mask of cells that don't express the genes.
        If None, uses all cells.

    Returns
    -------
    SoupChannel
        Modified SoupChannel with contamination_fraction set
    """

    if sc.soup_profile is None:
        raise ValueError("Must estimate soup profile first")

    # Find genes in both the gene list and our data
    available_genes = [g for g in non_expressed_genes if g in sc.gene_names]
    if not available_genes:
        raise ValueError("None of the specified genes found in data")

    print(f"Using {len(available_genes)} genes for contamination estimation: {available_genes}")

    # Get gene indices
    gene_indices = [np.where(sc.gene_names == g)[0][0] for g in available_genes]

    # Select cells to use for estimation
    if non_expressing_cells is None:
        cell_mask = np.ones(sc.n_cells, dtype=bool)
        print("Using all cells for contamination estimation")
    else:
        cell_mask = non_expressing_cells
        print(f"Using {np.sum(cell_mask)} cells for contamination estimation")

    # Calculate contamination fraction
    # Get observed counts for selected genes in selected cells
    observed_counts = sc.filtered_counts[gene_indices][:, cell_mask]
    total_observed = np.sum(observed_counts)

    # Get expected fraction from soup profile
    soup_fractions = sc.soup_profile.loc[available_genes, 'est'].values
    total_soup_fraction = np.sum(soup_fractions)

    # Calculate total UMIs in selected cells
    cell_umis = sc.metadata.loc[sc.filtered_barcodes[cell_mask], 'n_umis']
    total_umis = np.sum(cell_umis)

    # Contamination fraction = observed / (expected if pure soup)
    contamination_fraction = total_observed / (total_umis * total_soup_fraction)

    print(f"Estimated contamination fraction: {contamination_fraction:.3f} ({contamination_fraction * 100:.1f}%)")

    # Validate result
    if contamination_fraction > 1:
        raise ValueError(f"Contamination fraction > 1 ({contamination_fraction:.3f}). "
                         "This suggests the selected genes are actually expressed in the selected cells.")

    sc.set_contamination_fraction(contamination_fraction)

    return sc