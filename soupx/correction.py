"""
Count correction functions for removing background contamination.

Implements the core SoupX decontamination algorithms including
cluster-based correction and multinomial likelihood methods.
"""
import numpy as np
from scipy import sparse, stats
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .core import SoupChannel


def adjust_counts(
        sc: "SoupChannel",
        clusters: Optional[bool] = None,
        method: Literal["subtraction", "multinomial", "soupOnly"] = "subtraction",
        round_to_int: bool = False,
        verbose: bool = True
) -> sparse.csr_matrix:
    """
    Remove background contamination from count matrix.

    Implements multiple correction methods:
    1. 'subtraction': Simple subtraction (equation 5 from paper)
    2. 'multinomial': Maximum likelihood estimation
    3. 'soupOnly': P-value based removal

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with contamination fraction set
    clusters : bool or None, default None
        Whether to use cluster-based correction. If None, auto-detect from sc.clusters
    method : str, default "subtraction"
        Correction method to use
    round_to_int : bool, default False
        Whether to round results to integers using stochastic rounding
    verbose : bool, default True
        Print progress information

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

    # Auto-detect cluster usage
    if clusters is None:
        clusters = sc.clusters is not None

    if clusters and sc.clusters is None:
        raise ValueError("Clustering information not available. Use sc.set_clusters() or set clusters=False")

    if verbose:
        print(f"Adjusting counts using method '{method}' with contamination fraction: {sc.contamination_fraction:.3f}")
        if clusters:
            n_clusters = len(np.unique(sc.clusters[sc.clusters >= 0]))
            print(f"Using cluster-based correction with {n_clusters} clusters")

    # Apply cluster-based correction if requested
    if clusters:
        return _adjust_counts_clustered(sc, method, round_to_int, verbose)
    else:
        return _adjust_counts_single_cell(sc, method, round_to_int, verbose)


def _adjust_counts_single_cell(
        sc: "SoupChannel",
        method: str,
        round_to_int: bool,
        verbose: bool
) -> sparse.csr_matrix:
    """Apply correction at single-cell level."""

    if method == "subtraction":
        return _subtraction_method(sc, round_to_int, verbose)
    elif method == "multinomial":
        return _multinomial_method(sc, round_to_int, verbose)
    elif method == "soupOnly":
        return _soup_only_method(sc, round_to_int, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def _adjust_counts_clustered(
        sc: "SoupChannel",
        method: str,
        round_to_int: bool,
        verbose: bool
) -> sparse.csr_matrix:
    """
    Apply correction using cluster-based approach.

    Aggregates counts at cluster level, performs correction, then redistributes
    back to individual cells proportional to original cell sizes.
    """

    unique_clusters = np.unique(sc.clusters[sc.clusters >= 0])

    if verbose:
        print(f"Aggregating counts across {len(unique_clusters)} clusters")

    # Create cluster-aggregated count matrix
    cluster_counts = []
    cluster_metadata = []
    cluster_cell_mapping = {}

    for cluster_id in unique_clusters:
        cluster_mask = sc.clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        if len(cell_indices) == 0:
            continue

        # Aggregate counts for this cluster
        cluster_count_vector = np.array(sc.filtered_counts[:, cell_indices].sum(axis=1)).flatten()
        cluster_counts.append(cluster_count_vector)

        # Aggregate metadata
        cluster_umis = np.sum(sc.metadata.iloc[cell_indices]['n_umis'])
        cluster_rho = sc.contamination_fraction  # Same for all clusters in basic version

        cluster_metadata.append({
            'n_umis': cluster_umis,
            'rho': cluster_rho,
            'n_cells': len(cell_indices)
        })

        cluster_cell_mapping[cluster_id] = cell_indices

    # Create temporary cluster-level SoupChannel
    cluster_counts_matrix = sparse.csr_matrix(np.column_stack(cluster_counts))

    # Create temporary sc for cluster-level correction
    temp_sc = type(sc).__new__(type(sc))  # Create without calling __init__
    temp_sc.filtered_counts = cluster_counts_matrix
    temp_sc.soup_profile = sc.soup_profile
    temp_sc.contamination_fraction = sc.contamination_fraction
    temp_sc.n_genes = sc.n_genes
    temp_sc.n_cells = len(unique_clusters)
    temp_sc.gene_names = sc.gene_names
    temp_sc.clusters = None  # Don't use clustering for cluster-level correction

    # Create temporary metadata
    import pandas as pd
    temp_sc.metadata = pd.DataFrame(cluster_metadata,
                                   index=[f"cluster_{i}" for i in range(len(cluster_metadata))])

    # Apply correction at cluster level
    if verbose:
        print("Applying correction at cluster level")
    cluster_corrected = _adjust_counts_single_cell(temp_sc, method, False, verbose=False)

    # Redistribute back to single cells
    if verbose:
        print("Redistributing corrected counts back to individual cells")

    corrected_counts = sc.filtered_counts.copy().astype(float)

    for i, cluster_id in enumerate(unique_clusters):
        cell_indices = cluster_cell_mapping[cluster_id]

        if len(cell_indices) == 1:
            # Single cell cluster - direct assignment
            corrected_counts[:, cell_indices[0]] = cluster_corrected[:, i]
        else:
            # Multiple cells - redistribute proportionally
            original_cell_counts = sc.filtered_counts[:, cell_indices].toarray()
            cluster_corrected_counts = cluster_corrected[:, i].toarray().flatten()

            # Calculate proportional redistribution
            original_cluster_total = original_cell_counts.sum(axis=1)

            for cell_idx, orig_cell_idx in enumerate(cell_indices):
                for gene_idx in range(sc.n_genes):
                    if original_cluster_total[gene_idx] > 0:
                        proportion = original_cell_counts[gene_idx, cell_idx] / original_cluster_total[gene_idx]
                        corrected_counts[gene_idx, orig_cell_idx] = cluster_corrected_counts[gene_idx] * proportion
                    else:
                        corrected_counts[gene_idx, orig_cell_idx] = 0

    # Apply integer rounding if requested
    if round_to_int:
        corrected_counts = _stochastic_round(corrected_counts)

    # Convert back to sparse and ensure no negatives
    corrected_counts.data = np.maximum(corrected_counts.data, 0)
    corrected_counts.eliminate_zeros()

    return corrected_counts


def _subtraction_method(
        sc: "SoupChannel",
        round_to_int: bool,
        verbose: bool
) -> sparse.csr_matrix:
    """
    Simple subtraction method (equation 5 from SoupX paper).
    mg,c = ng,c - Nc*ρc*bg
    """

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
        corrected_counts = _stochastic_round(corrected_counts)

    # Report statistics
    if verbose:
        original_total = observed_counts.sum()
        corrected_total = corrected_counts.sum()
        removed_fraction = 1 - (corrected_total / original_total)
        print(f"Removed {removed_fraction:.1%} of total counts ({original_total - corrected_total:,} UMIs)")

    return corrected_counts


def _multinomial_method(
        sc: "SoupChannel",
        round_to_int: bool,
        verbose: bool
) -> sparse.csr_matrix:
    """
    Multinomial maximum likelihood method.

    More sophisticated than subtraction - explicitly models the multinomial
    distribution of contaminating counts.
    """

    if verbose:
        print("Using multinomial likelihood method")

    # Initialize with subtraction method
    if verbose:
        print("Initializing with subtraction method")
    initial_corrected = _subtraction_method(sc, round_to_int=False, verbose=False)

    observed_counts = sc.filtered_counts
    soup_fractions = sc.soup_profile['est'].values
    contamination_fraction = sc.contamination_fraction

    corrected_counts = initial_corrected.copy()

    if verbose:
        print(f"Optimizing multinomial likelihood for {sc.n_cells} cells")

    # Optimize each cell independently
    for cell_idx in range(sc.n_cells):
        if verbose and cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{sc.n_cells}")

        cell_umis = sc.metadata.iloc[cell_idx]['n_umis']
        expected_soup_umis = int(cell_umis * contamination_fraction)

        if expected_soup_umis == 0:
            continue

        # Get observed counts for this cell
        observed_cell = observed_counts[:, cell_idx].toarray().flatten()

        # Initialize with subtraction result
        corrected_cell = corrected_counts[:, cell_idx].toarray().flatten()

        # Iteratively optimize likelihood
        corrected_cell = _optimize_cell_multinomial(
            observed_cell, soup_fractions, expected_soup_umis, corrected_cell
        )

        # Update the matrix
        corrected_counts[:, cell_idx] = sparse.csr_matrix(corrected_cell.reshape(-1, 1))

    # Round to integers if requested
    if round_to_int:
        corrected_counts = _stochastic_round(corrected_counts)

    # Report statistics
    if verbose:
        original_total = observed_counts.sum()
        corrected_total = corrected_counts.sum()
        removed_fraction = 1 - (corrected_total / original_total)
        print(f"Multinomial method removed {removed_fraction:.1%} of total counts")

    return corrected_counts


def _optimize_cell_multinomial(
        observed: np.ndarray,
        soup_fractions: np.ndarray,
        target_soup_umis: int,
        initial_corrected: np.ndarray,
        max_iterations: int = 100
) -> np.ndarray:
    """
    Optimize multinomial likelihood for a single cell.

    This is a simplified version of the R implementation.
    """

    corrected = initial_corrected.copy()
    current_soup_umis = np.sum(observed - corrected)

    if current_soup_umis <= 0 or target_soup_umis <= 0:
        return corrected

    # Simple iterative approach
    for iteration in range(max_iterations):
        if abs(current_soup_umis - target_soup_umis) < 1:
            break

        if current_soup_umis < target_soup_umis:
            # Need to remove more counts - find genes to decrease
            gene_probs = soup_fractions * (corrected > 0)
            gene_probs = gene_probs / (np.sum(gene_probs) + 1e-10)

            # Select genes to decrease proportionally to soup profile
            genes_to_decrease = np.random.choice(
                len(gene_probs),
                size=min(target_soup_umis - current_soup_umis, int(np.sum(corrected))),
                p=gene_probs + 1e-10,
                replace=True
            )

            for gene_idx in genes_to_decrease:
                if corrected[gene_idx] > 0:
                    corrected[gene_idx] -= 1

        elif current_soup_umis > target_soup_umis:
            # Need to remove fewer counts - find genes to increase
            deficit = current_soup_umis - target_soup_umis

            # Increase counts for genes that were over-corrected
            over_corrected = observed - corrected
            total_over_corrected = np.sum(over_corrected)

            if total_over_corrected > 0:
                increase_probs = over_corrected / total_over_corrected
                genes_to_increase = np.random.choice(
                    len(increase_probs),
                    size=min(deficit, int(total_over_corrected)),
                    p=increase_probs + 1e-10,
                    replace=True
                )

                for gene_idx in genes_to_increase:
                    corrected[gene_idx] += 1

        current_soup_umis = np.sum(observed - corrected)

    # Ensure non-negative
    corrected = np.maximum(corrected, 0)

    return corrected


def _soup_only_method(
        sc: "SoupChannel",
        round_to_int: bool,
        verbose: bool,
        p_cut: float = 0.01
) -> sparse.csr_matrix:
    """
    SoupOnly method - uses p-values to determine which counts to remove.

    Removes counts for genes that cannot be confidently distinguished
    from background contamination.
    """

    if verbose:
        print(f"Using soupOnly method with p-value cutoff {p_cut}")

    observed_counts = sc.filtered_counts.copy()
    corrected_counts = observed_counts.copy().astype(float)

    soup_fractions = sc.soup_profile['est'].values
    contamination_fraction = sc.contamination_fraction

    for cell_idx in range(sc.n_cells):
        if verbose and cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{sc.n_cells}")

        cell_umis = sc.metadata.iloc[cell_idx]['n_umis']
        observed_cell = observed_counts[:, cell_idx].toarray().flatten()

        # Expected contamination for each gene
        expected_contamination = soup_fractions * cell_umis * contamination_fraction

        # Calculate p-values for each gene using Poisson test
        # H0: observed count = expected contamination count
        p_values = np.array([
            stats.poisson.cdf(obs, exp) if obs < exp
            else 1 - stats.poisson.cdf(obs - 1, exp)
            for obs, exp in zip(observed_cell, expected_contamination)
        ])

        # Remove counts for genes that cannot be distinguished from background
        soup_only_mask = p_values > p_cut
        corrected_cell = observed_cell.copy()
        corrected_cell[soup_only_mask] = 0  # Remove all expression for these genes

        corrected_counts[:, cell_idx] = sparse.csr_matrix(corrected_cell.reshape(-1, 1))

    # Round to integers if requested
    if round_to_int:
        corrected_counts = _stochastic_round(corrected_counts)

    # Remove zeros and ensure sparsity
    corrected_counts.eliminate_zeros()

    if verbose:
        original_total = observed_counts.sum()
        corrected_total = corrected_counts.sum()
        removed_fraction = 1 - (corrected_total / original_total)
        print(f"SoupOnly method removed {removed_fraction:.1%} of total counts")

    return corrected_counts


def _stochastic_round(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Apply stochastic rounding to convert float counts to integers.

    Takes floor of each count, then randomly rounds up with probability
    equal to the fractional part.
    """

    # Get fractional parts
    fractional_parts = counts.data - np.floor(counts.data)

    # Random decisions for rounding up
    round_up = np.random.random(len(fractional_parts)) < fractional_parts

    # Apply rounding
    counts.data = np.floor(counts.data) + round_up.astype(float)

    # Convert to integer type
    counts = counts.astype(int)

    # Remove any zeros that might have been created
    counts.eliminate_zeros()

    return counts