"""
SoupX Python Implementation

A Python port of the SoupX R package for removing ambient RNA contamination
from droplet-based single-cell RNA sequencing data.

Basic usage:
    import soupx

    # Create SoupChannel from raw and filtered count matrices
    sc = soupx.SoupChannel(raw_counts, filtered_counts)

    # Manual workflow
    sc.set_contamination_fraction(0.1)  # 10% contamination
    corrected_counts = soupx.adjust_counts(sc)

    # Automated workflow (requires clustering)
    sc.set_clusters(cluster_labels)
    sc = soupx.auto_est_cont(sc)  # Automatically estimate contamination
    corrected_counts = soupx.adjust_counts(sc)

For more advanced usage see individual module documentation.
"""

from .core import SoupChannel
from .estimation import estimate_soup, calculate_contamination_fraction, auto_est_cont, quickMarkers
from .correction import adjust_counts

__version__ = "0.2.0"

__all__ = [
    "SoupChannel",
    "estimate_soup",
    "calculate_contamination_fraction",
    "auto_est_cont",
    "quickMarkers",
    "adjust_counts",
]


# Convenience function for the most common workflow
def remove_ambient_rna(
        raw_counts,
        filtered_counts,
        contamination_fraction=None,
        non_expressed_genes=None,
        clusters=None,
        **kwargs
):
    """
    Convenience function for the standard SoupX workflow.

    Parameters
    ----------
    raw_counts : sparse matrix
        Raw droplet counts (genes x droplets)
    filtered_counts : sparse matrix
        Filtered cell counts (genes x cells)
    contamination_fraction : float, optional
        Manual contamination fraction (0-1). If None, will attempt automated estimation.
    non_expressed_genes : list, optional
        List of genes to use for contamination estimation (manual approach)
    clusters : array-like, optional
        Cluster assignments for automated contamination estimation
    **kwargs
        Additional arguments passed to SoupChannel constructor

    Returns
    -------
    sparse matrix
        Corrected count matrix
    """

    # Create SoupChannel
    sc = SoupChannel(raw_counts, filtered_counts, **kwargs)

    # Set clustering if provided
    if clusters is not None:
        sc.set_clusters(clusters)

    # Estimate contamination fraction
    if contamination_fraction is not None:
        # Manual contamination fraction
        sc.set_contamination_fraction(contamination_fraction)
    elif non_expressed_genes is not None:
        # Manual estimation using specific genes
        sc = calculate_contamination_fraction(sc, non_expressed_genes)
    elif clusters is not None:
        # Automated estimation using clustering
        sc = auto_est_cont(sc)
    else:
        raise ValueError(
            "Must provide either contamination_fraction, non_expressed_genes, or clusters"
        )

    # Correct counts
    return adjust_counts(sc)


# Convenience function for automated workflow
def auto_remove_ambient_rna(
        raw_counts,
        filtered_counts,
        clusters,
        **kwargs
):
    """
    Simplified convenience function for fully automated workflow.

    Parameters
    ----------
    raw_counts : sparse matrix
        Raw droplet counts (genes x droplets)
    filtered_counts : sparse matrix
        Filtered cell counts (genes x cells)
    clusters : array-like
        Cluster assignments for each cell
    **kwargs
        Additional arguments passed to SoupChannel constructor

    Returns
    -------
    tuple
        (corrected_counts, soup_channel) - corrected matrix and SoupChannel object
    """

    # Create SoupChannel
    sc = SoupChannel(raw_counts, filtered_counts, **kwargs)

    # Set clustering
    sc.set_clusters(clusters)

    # Automatically estimate contamination
    sc = auto_est_cont(sc)

    # Apply correction
    corrected_counts = adjust_counts(sc)

    return corrected_counts, sc