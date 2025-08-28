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


def _subtraction_method(sc, round_to_int=False, verbose=True, tol=1e-3, max_iterations=1000):
    """
    Iterative subtraction method that matches R SoupX exactly.

    This implements the same iterative algorithm as R SoupX to ensure
    exactly the target amount of contamination is removed.
    """

    # Get data in convenient form
    observed_counts = sc.filtered_counts.copy().astype(float)
    cell_umis = sc.metadata['n_umis'].values
    contamination_fraction = sc.contamination_fraction
    soup_fractions = sc.soup_profile['est'].values

    # Calculate target UMIs to remove per cell
    target_contamination_umis = cell_umis * contamination_fraction

    if verbose:
        print(f"Target contamination UMIs per cell: mean={target_contamination_umis.mean():.1f}")

    # Initialize output matrix
    corrected_counts = observed_counts.copy()

    # Convert to COO format for easier manipulation of sparse entries
    coo = corrected_counts.tocoo()

    # Initial subtraction - same as simple method
    for idx in range(len(coo.data)):
        gene_idx = coo.row[idx]
        cell_idx = coo.col[idx]

        # Expected contamination for this gene-cell pair
        expected_contamination = soup_fractions[gene_idx] * target_contamination_umis[cell_idx]

        # Subtract contamination (don't go below 0)
        coo.data[idx] = max(0, coo.data[idx] - expected_contamination)

    # Convert back to CSR for efficient column operations
    corrected_counts = coo.tocsr()

    # Iterative correction to hit targets exactly (like R)
    for iteration in range(max_iterations):
        # Calculate how much contamination we've actually removed per cell
        current_totals = np.array(corrected_counts.sum(axis=0)).flatten()
        original_totals = np.array(observed_counts.sum(axis=0)).flatten()
        removed_so_far = original_totals - current_totals

        # How much more do we need to remove per cell?
        still_to_remove = target_contamination_umis - removed_so_far

        # Check convergence
        max_error = np.max(np.abs(still_to_remove))
        if max_error < tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations (max error: {max_error:.6f})")
            break

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: max error = {max_error:.6f}")

        # Apply additional correction where needed
        coo = corrected_counts.tocoo()

        for idx in range(len(coo.data)):
            gene_idx = coo.row[idx]
            cell_idx = coo.col[idx]

            if abs(still_to_remove[cell_idx]) > tol and coo.data[idx] > 0:
                # Additional contamination to remove from this gene-cell pair
                additional_removal = soup_fractions[gene_idx] * still_to_remove[cell_idx]
                coo.data[idx] = max(0, coo.data[idx] - additional_removal)

        corrected_counts = coo.tocsr()

    else:
        if verbose:
            print(f"Warning: Did not converge after {max_iterations} iterations")

    # Final cleanup
    corrected_counts.eliminate_zeros()

    # Round to integers if requested
    if round_to_int:
        corrected_counts = _stochastic_round(corrected_counts)

    # Report final statistics
    if verbose:
        final_totals = np.array(corrected_counts.sum(axis=0)).flatten()
        final_removed = original_totals - final_totals
        actual_removal_fraction = np.sum(final_removed) / np.sum(original_totals)
        expected_removal_fraction = contamination_fraction

        print(f"Expected removal: {expected_removal_fraction:.1%}")
        print(f"Actual removal: {actual_removal_fraction:.1%}")
        print(f"Difference: {abs(actual_removal_fraction - expected_removal_fraction):.4f}")

    return corrected_counts


def _multinomial_method(
        sc: "SoupChannel",
        round_to_int: bool,
        verbose: bool
) -> sparse.csr_matrix:
    """
    Multinomial maximum likelihood method - faithful R implementation.

    Follows the R SoupX algorithm exactly:
    1. Initialize with subtraction method (rounded to int)
    2. For each cell, optimize soup counts using multinomial likelihood
    3. Return observed - optimized_soup_counts
    """
    if verbose:
        print("Using multinomial likelihood method")

    # Initialize with subtraction method, rounded to integers (as in R)
    if verbose:
        print("Initializing with subtraction method.")
    initial_subtracted = _subtraction_method(sc, round_to_int=True, verbose=False)

    # fitInit = observed - subtracted (these are the initial soup count estimates)
    fit_init = sc.filtered_counts - initial_subtracted

    observed_counts = sc.filtered_counts
    soup_probs = sc.soup_profile['est'].values  # ps in R code

    # Store results for each cell
    soup_counts_result = []

    if verbose:
        print(f"Fitting multinomial distribution to {sc.n_cells} cells/clusters.")

    # Loop over cells (as in R: for(i in seq(ncol(sc$toc))))
    for cell_idx in range(sc.n_cells):
        if verbose and cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{sc.n_cells}")

        # How many soup molecules do we expect for this cell?
        target_soup_umis = round(sc.contamination_fraction * sc.metadata.iloc[cell_idx]['n_umis'])

        # And what are the observational limits (lims in R)
        observed_cell = observed_counts[:, cell_idx].toarray().flatten()

        # Initialize with fitInit
        soup_counts = fit_init[:, cell_idx].toarray().flatten().astype(float)

        # Optimize this cell's soup counts
        soup_counts = _optimize_cell_multinomial_faithful(
            soup_counts, soup_probs, observed_cell, target_soup_umis, verbose=(verbose > 2)
        )

        soup_counts_result.append(soup_counts)

    # Convert results back to matrix format
    soup_counts_matrix = sparse.csr_matrix(np.column_stack(soup_counts_result))

    # Return corrected counts = observed - soup_counts
    corrected_counts = observed_counts - soup_counts_matrix

    # Ensure non-negative
    corrected_counts.data = np.maximum(corrected_counts.data, 0)
    corrected_counts.eliminate_zeros()

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


def _optimize_cell_multinomial_faithful(
        soup_counts: np.ndarray,
        soup_probs: np.ndarray,
        observed_limits: np.ndarray,
        target_soup_umis: int,
        verbose: bool = False,
        max_iterations: int = 10000
) -> np.ndarray:
    """
    Faithful implementation of R's multinomial likelihood optimization.

    This follows the R algorithm exactly:
    1. Calculate which genes can be increased/decreased
    2. Calculate likelihood gains for each move
    3. Make optimal moves until convergence
    """

    soup_counts = soup_counts.copy()

    for iteration in range(max_iterations):
        # Work out which we can increase/decrease
        increasable = soup_counts < observed_limits  # fit < lims
        decreasable = soup_counts > 0  # fit > 0

        if not np.any(increasable) and not np.any(decreasable):
            break

        # Calculate likelihood gains/costs for changing them
        # delInc = log(ps[increasable]) - log(fit[increasable]+1)
        delInc = np.full(len(soup_counts), -np.inf)
        if np.any(increasable):
            increasable_mask = increasable & (soup_probs > 0)
            if np.any(increasable_mask):
                delInc[increasable_mask] = (
                        np.log(soup_probs[increasable_mask]) -
                        np.log(soup_counts[increasable_mask] + 1)
                )

        # delDec = -log(ps[decreasable]) + log(fit[decreasable])
        delDec = np.full(len(soup_counts), -np.inf)
        if np.any(decreasable):
            decreasable_mask = decreasable & (soup_probs > 0)
            if np.any(decreasable_mask):
                delDec[decreasable_mask] = (
                        -np.log(soup_probs[decreasable_mask]) +
                        np.log(soup_counts[decreasable_mask])
                )

        # Find best moves
        max_delInc = np.max(delInc[np.isfinite(delInc)]) if np.any(np.isfinite(delInc)) else -np.inf
        max_delDec = np.max(delDec[np.isfinite(delDec)]) if np.any(np.isfinite(delDec)) else -np.inf

        wInc_all = np.where((delInc == max_delInc) & np.isfinite(delInc))[0]
        wDec_all = np.where((delDec == max_delDec) & np.isfinite(delDec))[0]

        if len(wInc_all) == 0 and len(wDec_all) == 0:
            break

        # Randomly select from ties (as in R)
        wInc = np.random.choice(wInc_all) if len(wInc_all) > 0 else None
        wDec = np.random.choice(wDec_all) if len(wDec_all) > 0 else None

        # How many soup counts do we currently have
        current_soup_umis = int(np.sum(soup_counts))

        if current_soup_umis < target_soup_umis:
            # Under-allocated, need to increase
            if verbose:
                print(f"Under-allocated ({current_soup_umis} of {target_soup_umis}), increasing...")
            if wInc is not None:
                soup_counts[wInc] += 1
        elif current_soup_umis > target_soup_umis:
            # Over-allocated, need to decrease
            if verbose:
                print(f"Over-allocated ({current_soup_umis} of {target_soup_umis}), decreasing...")
            if wDec is not None:
                soup_counts[wDec] -= 1
        else:
            # Exactly at target - check if swap improves likelihood
            delTot = max_delInc + max_delDec

            if verbose:
                print(f"At target ({current_soup_umis}), likelihood difference: {delTot}")

            if delTot == 0:
                # Ambiguous - distribute equally among tied options (as in R)
                if verbose:
                    print(
                        f"Ambiguous final configuration. Sharing {len(wDec_all)} reads between {len(np.unique(np.concatenate([wInc_all, wDec_all])))} equally likely options")

                # Take away from those that have them
                soup_counts[wDec_all] -= 1

                # Share equally among bucket
                zero_bucket = np.unique(np.concatenate([wInc_all, wDec_all]))
                soup_counts[zero_bucket] += len(wDec_all) / len(zero_bucket)
                break

            elif delTot < 0:
                # No improvement possible - minimum reached
                if verbose:
                    print("Unique final configuration.")
                break
            else:
                # Improvement possible - make the swap
                if wInc is not None and wDec is not None:
                    soup_counts[wInc] += 1
                    soup_counts[wDec] -= 1
                else:
                    break

    # Ensure non-negative and within bounds
    soup_counts = np.maximum(soup_counts, 0)
    soup_counts = np.minimum(soup_counts, observed_limits)

    return soup_counts


def _optimize_cell_multinomial(
        observed: np.ndarray,
        soup_fractions: np.ndarray,
        target_soup_umis: int,
        initial_corrected: np.ndarray,
        max_iterations: int = 100
) -> np.ndarray:
    """Optimize multinomial likelihood for a single cell."""

    corrected = initial_corrected.copy()
    current_soup_umis = np.sum(observed - corrected)

    if current_soup_umis <= 0 or target_soup_umis <= 0:
        return corrected

    # Ensure soup_fractions are valid probabilities
    if np.sum(soup_fractions) <= 0:
        return corrected

    soup_probs = soup_fractions / np.sum(soup_fractions)

    for iteration in range(max_iterations):
        if abs(current_soup_umis - target_soup_umis) < 1:
            break

        if current_soup_umis < target_soup_umis:
            # Need to remove more counts
            gene_probs = soup_probs * (corrected > 0)

            # Check if any genes can be decreased
            if np.sum(gene_probs) <= 0:
                break

            gene_probs = gene_probs / np.sum(gene_probs)

            # Ensure we have valid probabilities
            if np.any(np.isnan(gene_probs)) or np.sum(gene_probs) == 0:
                break

            genes_to_decrease = np.random.choice(
                len(gene_probs),
                size=min(target_soup_umis - current_soup_umis, int(np.sum(corrected))),
                p=gene_probs,
                replace=True
            )

            for gene_idx in genes_to_decrease:
                if corrected[gene_idx] > 0:
                    corrected[gene_idx] -= 1

        elif current_soup_umis > target_soup_umis:
            # Need to add back counts
            deficit = current_soup_umis - target_soup_umis
            over_corrected = observed - corrected
            total_over_corrected = np.sum(over_corrected)

            if total_over_corrected > 0:
                increase_probs = over_corrected / total_over_corrected

                # Ensure valid probabilities
                if np.sum(increase_probs) > 0:
                    genes_to_increase = np.random.choice(
                        len(increase_probs),
                        size=min(deficit, int(total_over_corrected)),
                        p=increase_probs,
                        replace=True
                    )

                    for gene_idx in genes_to_increase:
                        corrected[gene_idx] += 1

        current_soup_umis = np.sum(observed - corrected)

    return np.maximum(corrected, 0)


def _soup_only_method(
        sc: "SoupChannel",
        round_to_int: bool,
        verbose: bool,
        p_cut: float = 0.01
) -> sparse.csr_matrix:
    """
    Faithful implementation of R soupOnly method.

    Uses p-value based estimation with Fisher's method and chi-squared tests
    to identify genes that can be confidently identified as having endogenous
    expression and removes everything else.
    """

    if verbose:
        print("Identifying and removing genes likely to be pure contamination in each cell.")

    # Convert to COO format for easier manipulation (like R's dgTMatrix)
    observed_coo = sc.filtered_counts.tocoo()

    if verbose:
        print("Calculating probability of each gene being soup")

    # Calculate p-value against null of soup for each non-zero entry
    # p = ppois(out@x-1, expected_soup_counts, lower.tail=FALSE)
    p_values = []

    for idx in range(len(observed_coo.data)):
        gene_idx = observed_coo.row[idx]  # out@i in R (0-based)
        cell_idx = observed_coo.col[idx]  # out@j in R (0-based)
        count = observed_coo.data[idx]  # out@x in R

        # Expected soup count for this gene in this cell
        cell_umis = sc.metadata.iloc[cell_idx]['n_umis']
        soup_frac = sc.soup_profile.iloc[gene_idx]['est']
        contamination = sc.contamination_fraction
        expected_soup = cell_umis * soup_frac * contamination

        if expected_soup <= 0:
            p_val = 0.0  # No soup expected, so any count is significant
        else:
            # ppois(count-1, expected_soup, lower.tail=FALSE) = 1 - ppois(count-1, expected_soup)
            p_val = 1 - stats.poisson.cdf(count - 1, expected_soup)

        p_values.append(p_val)

    p_values = np.array(p_values)

    # Order by cell, then by p-value (as in R)
    # o = order(-(out@j+1), p, decreasing=TRUE)
    # This sorts first by cell (descending), then by p-value (descending)
    sort_keys = (-observed_coo.col, -p_values)  # Negative for descending order
    order_indices = np.lexsort(sort_keys)

    if verbose:
        print("Calculating probability of the next count being soup")

    # Get running totals for removal by cell
    # s = split(o, out@j[o]+1)
    # rTot = unlist(lapply(s,function(e) cumsum(out@x[e])), use.names=FALSE)
    running_totals = []
    cell_groups = {}

    # Group indices by cell
    for i, idx in enumerate(order_indices):
        cell_id = observed_coo.col[idx]
        if cell_id not in cell_groups:
            cell_groups[cell_id] = []
        cell_groups[cell_id].append(idx)

    # Calculate cumulative sums within each cell
    for cell_id in sorted(cell_groups.keys()):
        cell_indices = cell_groups[cell_id]
        cell_counts = [observed_coo.data[idx] for idx in cell_indices]
        cumsum_counts = np.cumsum(cell_counts)
        running_totals.extend(cumsum_counts)

    running_totals = np.array(running_totals)

    # Calculate soup probability vector for total counts removed
    # This is more complex in R - involves Poisson probabilities for total removal
    # For each entry, calculate probability that we should remove >= running_total counts
    soup_p_values = []

    current_pos = 0
    for cell_id in sorted(cell_groups.keys()):
        cell_indices = cell_groups[cell_id]
        cell_umis = sc.metadata.iloc[cell_id]['n_umis']
        expected_total_soup = cell_umis * sc.contamination_fraction

        for i, _ in enumerate(cell_indices):
            total_removed = running_totals[current_pos + i]

            # Probability of removing >= total_removed counts if pure soup
            if expected_total_soup <= 0:
                soup_p = 0.0
            else:
                soup_p = 1 - stats.poisson.cdf(total_removed - 1, expected_total_soup)

            soup_p_values.append(soup_p)

        current_pos += len(cell_indices)

    soup_p_values = np.array(soup_p_values)

    # Combine p-values using Fisher's method
    # chi_squared = -2 * (log(p1) + log(p2))
    # Combined p-value from chi-squared distribution with 4 degrees of freedom
    combined_p_values = []

    for i in range(len(p_values)):
        idx = order_indices[i]
        gene_p = p_values[idx]
        soup_p = soup_p_values[i]

        # Avoid log(0) issues
        gene_p = max(gene_p, 1e-300)
        soup_p = max(soup_p, 1e-300)

        # Fisher's method: chi_squared = -2 * sum(log(p_values))
        chi_squared = -2 * (np.log(gene_p) + np.log(soup_p))

        # Combined p-value from chi-squared distribution with 4 degrees of freedom
        combined_p = 1 - stats.chi2.cdf(chi_squared, df=4)
        combined_p_values.append(combined_p)

    combined_p_values = np.array(combined_p_values)

    # Apply cutoff and determine which counts to remove
    if verbose:
        print(f"Applying p-value cutoff of {p_cut}")

    # Create output matrix - start with original
    corrected_counts = sc.filtered_counts.copy().astype(float)

    # Remove counts where combined p-value > p_cut (insufficient evidence of endogenous expression)
    for i, idx in enumerate(order_indices):
        gene_idx = observed_coo.row[idx]
        cell_idx = observed_coo.col[idx]

        if combined_p_values[i] > p_cut:
            # Remove this count - set to 0
            corrected_counts[gene_idx, cell_idx] = 0

    # Ensure sparsity
    corrected_counts.eliminate_zeros()

    # Round to integers if requested
    if round_to_int:
        corrected_counts = _stochastic_round(corrected_counts)

    # Report statistics
    if verbose:
        original_total = sc.filtered_counts.sum()
        corrected_total = corrected_counts.sum()
        removed_fraction = 1 - (corrected_total / original_total)
        print(f"SoupOnly method removed {removed_fraction:.1%} of total counts")

        # Report how many genes/cells were affected
        n_removed = np.sum(combined_p_values > p_cut)
        print(f"Removed {n_removed}/{len(combined_p_values)} gene-cell combinations")

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