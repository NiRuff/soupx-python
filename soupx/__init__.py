"""
SoupX Python Implementation

A Python port of the SoupX R package for removing ambient RNA contamination
from droplet-based single-cell RNA sequencing data.

Basic usage:
    import soupx

    # Create SoupChannel from raw and filtered count matrices
    sc = soupx.SoupChannel(raw_counts, filtered_counts)

    # Set contamination fraction (manual approach)
    sc.set_contamination_fraction(0.1)  # 10% contamination

    # Remove contamination
    corrected_counts = soupx.adjust_counts(sc)

For more advanced usage see individual module documentation.
"""

from .core import SoupChannel
from .estimation import estimate_soup, calculate_contamination_fraction
from .correction import adjust_counts

__version__ = "0.1.0"

__all__ = [
    "SoupChannel",
    "estimate_soup",
    "calculate_contamination_fraction",
    "adjust_counts",
]


# Convenience function for the most common workflow
def remove_ambient_rna(
        raw_counts,
        filtered_counts,
        contamination_fraction=None,
        non_expressed_genes=None,
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
        Manual contamination fraction (0-1). If None, must provide non_expressed_genes.
    non_expressed_genes : list, optional
        List of genes to use for contamination estimation
    **kwargs
        Additional arguments passed to SoupChannel constructor

    Returns
    -------
    sparse matrix
        Corrected count matrix
    """

    # Create SoupChannel
    sc = SoupChannel(raw_counts, filtered_counts, **kwargs)

    # Set or estimate contamination fraction
    if contamination_fraction is not None:
        sc.set_contamination_fraction(contamination_fraction)
    elif non_expressed_genes is not None:
        sc = calculate_contamination_fraction(sc, non_expressed_genes)
    else:
        raise ValueError("Must provide either contamination_fraction or non_expressed_genes")

    # Correct counts
    return adjust_counts(sc)