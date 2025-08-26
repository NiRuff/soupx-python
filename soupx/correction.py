"""
Count correction functions for removing background contamination.

Implements the core SoupX decontamination algorithm.
"""
import numpy as np
from scipy import sparse
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core import SoupChannel


def adjust_counts(
        sc: "SoupChannel",
        method: Literal["subtraction"] = "subtraction",
        round_to_int: bool = False
) -> sparse.csr_matrix:
    """
    Remove background contamination from count matrix.

    Implements equation (5) from SoupX paper:
    mg,c = ng,c - Nc*ρc*bg

    Where:
    - mg,c = corrected counts for gene g in cell c
    - ng,c = observed counts
    - Nc = total UMIs in cell c
    - ρc = contamination fraction for cell c
    - bg = soup fraction for gene g

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with contamination fraction set
    method : str, default "subtraction"
        Correction method. Only "subtraction" implemented for now.
    round_to_int : bool, default False
        Whether to round results to integers

    Returns
    -------
    sparse.csr_matrix
        Corrected count matrix (genes x cells)
    """

    # Validation
    if sc.soup_profile is None:
        raise ValueError("Must estimate soup profile first")
    if sc.contamination_fraction is None:
        raise ValueError("Must set contamination fraction first")
    if method != "subtraction":
        raise NotImplementedError(f"Method '{method}' not implemented")

    print(f"Adjusting counts using contamination fraction: {sc.contamination_fraction:.3f}")

    # Get data in convenient form
    observed_counts = sc.filtered_counts.copy()  # ng,c
    cell_umis = sc.metadata['n_umis'].values  # Nc
    contamination_fraction = sc.contamination_fraction  # ρc (global)
    soup_fractions = sc.soup_profile['est'].values  # bg

    # Calculate expected contamination counts for each gene in each cell
    # Expected contamination = Nc * ρc * bg
    expected_contamination = np.outer(soup_fractions, cell_umis * contamination_fraction)

    # Convert to sparse matrix for efficient subtraction
    expected_contamination_sparse = sparse.csr_matrix(expected_contamination)

    # Subtract contamination: mg,c = ng,c - Nc*ρc*bg
    corrected_counts = observed_counts - expected_contamination_sparse

    # Ensure no negative counts
    corrected_counts.data = np.maximum(corrected_counts.data, 0)

    # Remove explicit zeros to maintain sparsity
    corrected_counts.eliminate_zeros()

    # Round to integers if requested
    if round_to_int:
        corrected_counts.data = np.round(corrected_counts.data)
        corrected_counts = corrected_counts.astype(int)

    # Report statistics
    original_total = observed_counts.sum()
    corrected_total = corrected_counts.sum()
    removed_fraction = 1 - (corrected_total / original_total)

    print(f"Removed {removed_fraction:.1%} of total counts ({original_total - corrected_total:,} UMIs)")

    return corrected_counts


def _iterative_subtraction(
        observed_counts: sparse.csr_matrix,
        soup_fractions: np.ndarray,
        expected_soup_counts: np.ndarray,
        tolerance: float = 1e-3,
        max_iterations: int = 100
) -> sparse.csr_matrix:
    """
    Iterative subtraction method (more accurate but slower).

    This implements the more sophisticated approach from the R version
    where contamination is removed iteratively to handle the redistribution
    of "unused" counts.

    Currently not used in main adjust_counts but kept for potential future use.
    """

    corrected = observed_counts.copy().astype(float)

    for iteration in range(max_iterations):
        # Calculate how many counts we still need to remove per cell
        current_totals = np.array(corrected.sum(axis=0)).flatten()
        target_totals = current_totals - expected_soup_counts

        # If we're close enough, stop
        excess = current_totals - target_totals
        if np.max(excess) < tolerance:
            break

        # Remove counts proportional to soup profile
        # This is a simplified version - the R version is more complex
        for cell_idx in range(corrected.shape[1]):
            if excess[cell_idx] > tolerance:
                # Calculate removal for this cell
                cell_column = corrected[:, cell_idx].toarray().flatten()
                removal = excess[cell_idx] * soup_fractions

                # Don't remove more than we have
                removal = np.minimum(removal, cell_column)

                # Apply removal
                new_values = cell_column - removal
                corrected[:, cell_idx] = sparse.csr_matrix(new_values.reshape(-1, 1))

    # Ensure no negative values
    corrected.data = np.maximum(corrected.data, 0)
    corrected.eliminate_zeros()

    return corrected