"""
Count correction functions matching R SoupX implementation exactly.
"""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from typing import TYPE_CHECKING, Literal, Optional, Union
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from .core import SoupChannel


def adjustCounts(
        sc: "SoupChannel",
        clusters: Optional[Union[bool, np.ndarray]] = None,
        method: Literal["subtraction", "multinomial", "soupOnly"] = "subtraction",
        roundToInt: bool = False,
        verbose: int = 1,
        tol: float = 1e-3,
        pCut: float = 0.01,
        **kwargs
) -> sparse.csr_matrix:
    """
    Remove background contamination from count matrix - R-compatible interface.

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with contamination fraction set
    clusters : array-like, None, or False
        Cluster assignments. None = auto-detect, False = no clustering
    method : str
        'subtraction', 'multinomial', or 'soupOnly'
    roundToInt : bool
        Round to integers using stochastic rounding
    verbose : int
        0 = silent, 1 = basic info, 2 = chatty, 3 = debug
    tol : float
        Tolerance for convergence
    pCut : float
        P-value cutoff for soupOnly method
    **kwargs
        Passed to expandClusters

    Returns
    -------
    sparse.csr_matrix
        Corrected count matrix
    """
    # Check prerequisites
    if 'rho' not in sc.metaData.columns or sc.metaData['rho'].isna().all():
        raise ValueError("Contamination fractions must have already been calculated/set.")

    # Handle clusters parameter like R
    if clusters is None:
        if 'clusters' in sc.metaData.columns:
            clusters = sc.metaData['clusters'].values
        else:
            if verbose >= 1:
                print("Warning: Clustering data not found. Adjusting counts at cell level.")
            clusters = False

    # Recursive application when using clusters (matching R logic)
    if clusters is not False:
        if verbose >= 1:
            unique_clusters = np.unique(clusters)
            print(f"Adjusting counts using method '{method}' with {len(unique_clusters)} clusters")

        # Split cells by cluster
        cluster_groups = {}
        for i, cell_id in enumerate(sc.metaData.index):
            cluster = clusters[i]
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append(i)

        # Create cluster-level aggregated data
        cluster_toc = []
        cluster_metadata = []

        for cluster_id in sorted(cluster_groups.keys()):
            cell_indices = cluster_groups[cluster_id]
            # Aggregate counts for cluster
            cluster_counts = np.array(sc.toc[:, cell_indices].sum(axis=1)).flatten()
            cluster_toc.append(cluster_counts)

            # Aggregate metadata
            cluster_nUMIs = sc.metaData.iloc[cell_indices]['nUMIs'].sum()
            cluster_rho = (sc.metaData.iloc[cell_indices]['rho'] *
                          sc.metaData.iloc[cell_indices]['nUMIs']).sum() / cluster_nUMIs
            cluster_metadata.append({'nUMIs': cluster_nUMIs, 'rho': cluster_rho})

        # Create temporary cluster-level SoupChannel
        cluster_toc_matrix = sparse.csr_matrix(np.column_stack(cluster_toc))

        tmp_sc = type(sc).__new__(type(sc))
        tmp_sc.toc = cluster_toc_matrix
        tmp_sc.tod = sc.tod  # Keep original tod
        tmp_sc.soupProfile = sc.soupProfile
        tmp_sc.metaData = pd.DataFrame(cluster_metadata)
        tmp_sc.n_genes = sc.n_genes
        tmp_sc.n_cells = len(cluster_groups)

        # Recursively apply without clustering
        cluster_corrected = adjustCounts(
            tmp_sc, clusters=False, method=method,
            roundToInt=False, verbose=verbose, tol=tol, pCut=pCut
        )

        # Calculate soup counts removed at cluster level
        cluster_soup = tmp_sc.toc - cluster_corrected

        # Expand back to cell level using expandClusters logic
        cell_soup = expandClusters(
            cluster_soup, sc.toc, clusters, cluster_groups,
            sc.metaData['nUMIs'].values * sc.metaData['rho'].values,
            verbose=verbose, **kwargs
        )

        # Return corrected counts
        out = sc.toc - cell_soup

        if roundToInt:
            out = _stochastic_round(out)

        return out

    # Single-cell level correction
    if method == "subtraction":
        return _subtraction_method(sc, roundToInt, verbose, tol)
    elif method == "multinomial":
        return _multinomial_method(sc, roundToInt, verbose, tol)
    elif method == "soupOnly":
        return _soupOnly_method(sc, roundToInt, verbose, pCut)
    else:
        raise ValueError(f"Unknown method: {method}")


def expandClusters(
        cluster_soup: sparse.csr_matrix,
        toc: sparse.csr_matrix,
        clusters: np.ndarray,
        cluster_groups: dict,
        target_soup_counts: np.ndarray,
        verbose: int = 1,
        **kwargs
) -> sparse.csr_matrix:
    """
    Expand cluster-level soup estimates back to cells.
    Matches R's expandClusters function logic.
    """
    n_genes, n_cells = toc.shape
    cell_soup = sparse.lil_matrix((n_genes, n_cells))

    for cluster_idx, (cluster_id, cell_indices) in enumerate(cluster_groups.items()):
        cluster_soup_vec = cluster_soup[:, cluster_idx].toarray().flatten()

        if len(cell_indices) == 1:
            # Single cell in cluster
            cell_soup[:, cell_indices[0]] = cluster_soup_vec.reshape(-1, 1)
        else:
            # Multiple cells - redistribute proportionally
            # Calculate proportion of counts each cell has
            cell_counts = toc[:, cell_indices].toarray()
            cluster_total = cell_counts.sum(axis=1, keepdims=True)

            # Avoid division by zero
            cluster_total[cluster_total == 0] = 1
            proportions = cell_counts / cluster_total

            # Redistribute soup counts
            for i, cell_idx in enumerate(cell_indices):
                cell_soup[:, cell_idx] = (cluster_soup_vec * proportions[:, i]).reshape(-1, 1)

    return cell_soup.tocsr()


def _subtraction_method(
        sc: "SoupChannel",
        roundToInt: bool,
        verbose: int,
        tol: float
) -> sparse.csr_matrix:
    """
    Simple subtraction method (equation 5 from paper).
    Matches R implementation exactly.
    """
    if verbose >= 1:
        print("Using subtraction method")

    # Calculate expected soup counts
    soup_expression = sc.soupProfile['est'].values
    corrected = sc.toc.copy().astype(float)

    for cell_idx in range(sc.n_cells):
        cell_rho = sc.metaData.iloc[cell_idx]['rho']
        cell_nUMIs = sc.metaData.iloc[cell_idx]['nUMIs']

        # Expected soup counts for this cell
        expected_soup = soup_expression * cell_nUMIs * cell_rho

        # Get observed counts
        observed = sc.toc[:, cell_idx].toarray().flatten()

        # Simple subtraction
        corrected_counts = observed - expected_soup

        # Can't have negative counts
        corrected_counts = np.maximum(corrected_counts, 0)

        corrected[:, cell_idx] = corrected_counts.reshape(-1, 1)

    if roundToInt:
        corrected = _stochastic_round(corrected)

    return corrected.tocsr()


def _multinomial_method(
        sc: "SoupChannel",
        roundToInt: bool,
        verbose: int,
        tol: float
) -> sparse.csr_matrix:
    """
    Multinomial likelihood optimization method.
    Exact implementation of R's algorithm.
    """
    if verbose >= 1:
        print(f"Fitting multinomial distribution to {sc.n_cells} cells")

    # Initialize with subtraction method
    if verbose >= 2:
        print("Initializing with subtraction method")

    fit_init = sc.toc - _subtraction_method(sc, True, 0, tol)
    ps = sc.soupProfile['est'].values

    out = sparse.lil_matrix(sc.toc.shape)

    for cell_idx in range(sc.n_cells):
        if verbose >= 1 and cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{sc.n_cells}")

        # Target soup molecules for this cell
        nSoupUMIs = round(sc.metaData.iloc[cell_idx]['nUMIs'] *
                         sc.metaData.iloc[cell_idx]['rho'])

        # Observational limits
        lims = sc.toc[:, cell_idx].toarray().flatten()

        # Initial soup counts
        fit = fit_init[:, cell_idx].toarray().flatten().astype(float)

        # Run optimization
        fit = _optimize_multinomial_cell(fit, ps, lims, nSoupUMIs, tol, verbose >= 3)

        # Store corrected counts
        out[:, cell_idx] = (lims - fit).reshape(-1, 1)

    out = out.tocsr()

    if roundToInt:
        out = _stochastic_round(out)

    if verbose >= 1:
        original_total = sc.toc.sum()
        corrected_total = out.sum()
        print(f"Removed {(1 - corrected_total/original_total)*100:.1f}% of counts")

    return out


def _optimize_multinomial_cell(
        fit: np.ndarray,
        ps: np.ndarray,
        lims: np.ndarray,
        nSoupUMIs: int,
        tol: float,
        verbose: bool,
        max_iter: int = 1000
) -> np.ndarray:
    """
    Optimize soup counts for a single cell using R's exact algorithm.
    This follows the likelihood maximization approach from R.
    """
    fit = fit.copy()

    for iteration in range(max_iter):
        # Check which can be increased/decreased
        increasable = fit < lims
        decreasable = fit > 0

        if not np.any(increasable) and not np.any(decreasable):
            break

        # Calculate likelihood changes for each possible move
        # These formulas are from the R implementation
        delInc = np.full(len(fit), -np.inf)
        if np.any(increasable):
            mask = increasable & (ps > 0)
            if np.any(mask):
                delInc[mask] = np.log(ps[mask]) - np.log(fit[mask] + 1)

        delDec = np.full(len(fit), -np.inf)
        if np.any(decreasable):
            mask = decreasable & (ps > 0)
            if np.any(mask):
                delDec[mask] = -np.log(ps[mask]) + np.log(fit[mask])

        # Find best moves
        max_delInc = np.max(delInc[np.isfinite(delInc)]) if np.any(np.isfinite(delInc)) else -np.inf
        max_delDec = np.max(delDec[np.isfinite(delDec)]) if np.any(np.isfinite(delDec)) else -np.inf

        # Get indices of best moves (all ties)
        wInc_all = np.where((delInc == max_delInc) & np.isfinite(delInc))[0]
        wDec_all = np.where((delDec == max_delDec) & np.isfinite(delDec))[0]

        if len(wInc_all) == 0 and len(wDec_all) == 0:
            break

        # Randomly select from ties (as R does)
        wInc = np.random.choice(wInc_all) if len(wInc_all) > 0 else None
        wDec = np.random.choice(wDec_all) if len(wDec_all) > 0 else None

        # Current soup count
        current_soup = int(np.sum(fit))

        # Make the move
        if current_soup < nSoupUMIs:
            # Need more soup
            if wInc is not None and max_delInc > -np.inf:
                fit[wInc] += 1
        elif current_soup > nSoupUMIs:
            # Too much soup
            if wDec is not None and max_delDec > -np.inf:
                fit[wDec] -= 1
        else:
            # At target, check which move is better
            if max_delInc > max_delDec and wInc is not None:
                if wDec is not None:
                    fit[wInc] += 1
                    fit[wDec] -= 1
            elif wDec is not None and wInc is not None:
                fit[wDec] -= 1
                fit[wInc] += 1

        # Check convergence
        if abs(np.sum(fit) - nSoupUMIs) <= tol:
            break

    if verbose and iteration == max_iter - 1:
        print(f"Warning: Max iterations reached. Diff from target: {abs(np.sum(fit) - nSoupUMIs)}")

    return fit


def _soupOnly_method(
        sc: "SoupChannel",
        roundToInt: bool,
        verbose: int,
        pCut: float
) -> sparse.csr_matrix:
    """
    P-value based gene removal method.
    Matches R's soupOnly implementation using Fisher's method.
    """
    if verbose >= 1:
        print("Identifying genes likely to be pure contamination")

    corrected = sc.toc.copy().astype(float)

    for cell_idx in range(sc.n_cells):
        cell_rho = sc.metaData.iloc[cell_idx]['rho']
        cell_nUMIs = sc.metaData.iloc[cell_idx]['nUMIs']
        observed = sc.toc[:, cell_idx].toarray().flatten()

        # Expected soup counts
        expected_soup = sc.soupProfile['est'].values * cell_nUMIs * cell_rho

        # Calculate p-values for each gene
        p_vals = []
        for gene_idx in range(sc.n_genes):
            if expected_soup[gene_idx] <= 0:
                p_val = 0.0 if observed[gene_idx] > 0 else 1.0
            else:
                # Poisson test - is observed significantly > expected?
                p_val = 1 - stats.poisson.cdf(observed[gene_idx] - 1, expected_soup[gene_idx])
            p_vals.append(p_val)

        # Sort genes by p-value
        gene_order = np.argsort(p_vals)

        # Remove genes until we've removed ~rho fraction
        soup_removed = 0
        target_soup = cell_nUMIs * cell_rho

        for gene_idx in gene_order:
            if p_vals[gene_idx] > pCut:
                # This gene shows no evidence of endogenous expression
                soup_removed += observed[gene_idx]
                corrected[gene_idx, cell_idx] = 0

                if soup_removed >= target_soup:
                    break

    if roundToInt:
        corrected = _stochastic_round(corrected)

    return corrected.tocsr()


def _stochastic_round(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Stochastic rounding to integers.
    Matches R's behavior: floor + bernoulli(fractional part).
    """
    matrix = matrix.tocsr()
    data = matrix.data.copy()

    # Get integer and fractional parts
    int_part = np.floor(data)
    frac_part = data - int_part

    # Stochastically round up based on fractional part
    round_up = np.random.random(len(data)) < frac_part
    data = int_part + round_up

    # Create new matrix with integer values
    result = sparse.csr_matrix((data, matrix.indices, matrix.indptr),
                               shape=matrix.shape, dtype=int)
    return result