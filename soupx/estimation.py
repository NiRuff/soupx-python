"""
Contamination estimation functions matching R SoupX exactly.
"""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SoupChannel


def autoEstCont(
        sc: "SoupChannel",
        topMarkers: Optional[pd.DataFrame] = None,
        tfidfMin: float = 1.0,
        soupQuantile: float = 0.90,
        maxMarkers: int = 100,
        contaminationRange: Tuple[float, float] = (0.01, 0.8),
        rhoMaxFDR: float = 0.2,
        priorRho: float = 0.05,
        priorRhoStdDev: float = 0.10,
        doPlot: bool = True,
        forceAccept: bool = False,
        verbose: bool = True
) -> "SoupChannel":
    """
    Automatically estimate contamination fraction - R-compatible interface.

    Exact implementation of R's autoEstCont algorithm:
    1. Collapse to cluster level
    2. Find markers using quickMarkers
    3. Apply tfidf and soup quantile filters
    4. Use estimateNonExpressingCells for validation
    5. Calculate posterior distributions and aggregate

    Parameters match R function exactly.
    """
    # Validation
    if 'clusters' not in sc.metaData.columns:
        raise ValueError("Clustering information must be supplied, run setClusters first.")

    if sc.soupProfile is None:
        raise ValueError("Soup profile must be calculated first.")

    # Collapse by cluster (matching R: do.call(cbind,lapply(...)))
    clusters = sc.metaData['clusters'].values
    unique_clusters = np.unique(clusters)

    # Create cluster-aggregated matrix
    cluster_toc = []
    cluster_metadata = []

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        # Aggregate counts
        cluster_counts = np.array(sc.toc[:, cell_indices].sum(axis=1)).flatten()
        cluster_toc.append(cluster_counts)

        # Aggregate metadata
        cluster_nUMIs = sc.metaData.iloc[cell_indices]['nUMIs'].sum()
        cluster_metadata.append({'nUMIs': cluster_nUMIs})

    cluster_toc_matrix = np.column_stack(cluster_toc)

    # Get soup profile ordered by expression
    soupProf = sc.soupProfile.sort_values('est', ascending=False)
    soupMin = np.quantile(soupProf['est'].values, soupQuantile)
    tgts = soupProf.index[soupProf['est'] > soupMin].tolist()  # Define tgts here

    # Find or use provided markers
    if topMarkers is None:
        # Get markers using quickMarkers with gene names
        gene_names = sc.gene_names if hasattr(sc, 'gene_names') else list(sc.soupProfile.index)
        mrks = quickMarkers(sc.toc, clusters, N=None, verbose=verbose,
                            gene_names=gene_names, expressCut=0.9)  # Pass expressCut=0.9
        # Keep only most specific entry per gene
        mrks = mrks.sort_values(['gene', 'tfidf'], ascending=[True, False])
        mrks = mrks[~mrks.duplicated(subset='gene', keep='first')]
        # Order by tfidf
        mrks = mrks.sort_values('tfidf', ascending=False)
        # Apply tfidf cutoff
        mrks = mrks[mrks['tfidf'] > tfidfMin]
        print("Top 20 markers after sorting and filtering:")
        print(mrks.head(20))
    else:
        mrks = topMarkers

    # Filter to genes highly expressed in soup
    filtPass = mrks[mrks['gene'].isin(tgts)]
    tgts = filtPass.head(maxMarkers)['gene'].tolist()
    print("\nGenes passing soup quantile filter:")
    print(tgts[:20])
    print("\nNumber of genes passing both filters:")
    print(len(filtPass))

    if verbose:
        print(f"{len(mrks)} genes passed tf-idf cut-off and {len(filtPass)} soup quantile filter. "
              f"Taking the top {len(tgts)}.")

    if len(tgts) == 0:
        raise ValueError("No plausible marker genes found. Reduce tfidfMin or soupQuantile")

    if len(tgts) < 10:
        print(f"Warning: Fewer than 10 marker genes found ({len(tgts)}). "
              "Is this channel low complexity?")

    # Estimate contamination for each cluster/gene pair
    estimates = []

    for gene in tgts:
        gene_idx = sc.soupProfile.index.get_loc(gene)
        soup_expression = sc.soupProfile.iloc[gene_idx]['est']

        # Find which cluster this is a marker for
        marker_info = mrks[mrks['gene'] == gene].iloc[0]
        marker_cluster = marker_info['cluster']

        # Use estimateNonExpressingCells to find valid clusters
        non_expressing = estimateNonExpressingCells(
            sc, [gene], clusters,
            maximumContamination=contaminationRange[1],
            FDR=rhoMaxFDR,
            verbose=False
        )

        # Calculate contamination estimates for non-expressing clusters
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            if cluster_id == marker_cluster:
                continue  # Skip the cluster where this is a marker

            cluster_mask = clusters == cluster_id
            if not np.any(non_expressing[cluster_mask]):
                continue  # Skip if not confidently non-expressing

            # Get cluster counts
            cluster_counts = cluster_toc_matrix[:, cluster_idx]
            observed = cluster_counts[gene_idx]
            total_umis = np.sum(cluster_counts)

            if total_umis == 0 or soup_expression == 0:
                continue

            # Calculate MLE estimate of rho
            rho_est = observed / (soup_expression * total_umis)

            # Apply contamination range limits
            rho_est = np.clip(rho_est, contaminationRange[0], contaminationRange[1])

            # Calculate posterior using gamma prior
            # Shape and rate parameters from prior mean and std dev
            prior_shape = (priorRho / priorRhoStdDev) ** 2
            prior_rate = priorRho / (priorRhoStdDev ** 2)

            # Posterior parameters (conjugate prior)
            post_shape = prior_shape + observed
            post_rate = prior_rate + soup_expression * total_umis

            # Store estimate with posterior parameters
            estimates.append({
                'gene': gene,
                'cluster': cluster_id,
                'rho_est': rho_est,
                'post_shape': post_shape,
                'post_rate': post_rate,
                'observed': observed,
                'total_umis': total_umis
            })

    if len(estimates) == 0:
        raise ValueError("No valid contamination estimates. Check your data and parameters.")

    # Aggregate estimates to get global contamination
    # Create density estimate from all posteriors
    rho_range = np.linspace(contaminationRange[0], contaminationRange[1], 1000)
    total_density = np.zeros_like(rho_range)

    for est in estimates:
        # Add this estimate's posterior to total
        density = stats.gamma.pdf(
            rho_range,
            a=est['post_shape'],
            scale=1/est['post_rate']
        )
        total_density += density

    # Find mode of aggregated distribution
    global_rho = rho_range[np.argmax(total_density)]

    if verbose:
        print(f"Estimated global contamination fraction: {global_rho:.3f}")

    # Set contamination fraction
    sc.set_contamination_fraction(global_rho, forceAccept=forceAccept)

    # Store fit information
    sc.fit = {
        'estimates': pd.DataFrame(estimates),
        'global_rho': global_rho,
        'rho_range': rho_range,
        'density': total_density
    }

    if doPlot and verbose:
        # Could add plotting here if matplotlib available
        pass

    return sc


def estimateNonExpressingCells(
        sc: "SoupChannel",
        nonExpressedGeneList: list,
        clusters: Optional[np.ndarray] = None,
        maximumContamination: float = 0.8,
        FDR: float = 0.2,
        verbose: bool = True
) -> np.ndarray:
    """
    Estimate which cells don't express specific genes.
    Matches R's estimateNonExpressingCells using Fisher's method.

    Returns boolean array indicating cells confidently not expressing genes.
    """
    if clusters is None:
        if 'clusters' in sc.metaData.columns:
            clusters = sc.metaData['clusters'].values
        else:
            clusters = np.array(['0'] * sc.n_cells)

    unique_clusters = np.unique(clusters)
    use_cells = np.ones(sc.n_cells, dtype=bool)

    # Get gene indices
    gene_indices = []
    for gene in nonExpressedGeneList:
        try:
            if hasattr(sc.soupProfile, 'index'):
                gene_idx = sc.soupProfile.index.get_loc(gene)
                gene_indices.append(gene_idx)
            else:
                # If no index, assume gene is an integer index
                gene_indices.append(int(gene))
        except (KeyError, ValueError):
            if verbose:
                print(f"Warning: Gene {gene} not found in data")
            continue

    if len(gene_indices) == 0:
        if verbose:
            print("Warning: None of the specified genes found in data")
        return use_cells

    # Test each cluster
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        if len(cell_indices) == 0:
            continue

        cluster_passes = True

        for gene_idx in gene_indices:
            # Get observed counts for this gene in cluster
            obs_counts = sc.toc[gene_idx, cell_indices].toarray().flatten()

            # Calculate expected counts under maximum contamination
            soup_expression = sc.soupProfile.iloc[gene_idx]['est']
            cell_umis = sc.metaData.iloc[cell_indices]['nUMIs'].values
            expected_counts = soup_expression * cell_umis * maximumContamination

            # Calculate p-values using Poisson test
            p_values = []
            for obs, exp in zip(obs_counts, expected_counts):
                if exp <= 0:
                    p_val = 0.0 if obs > 0 else 1.0
                else:
                    # Test if observed > expected (evidence of expression)
                    p_val = 1 - stats.poisson.cdf(obs - 1, exp)
                p_values.append(p_val)

            # Combine p-values using Fisher's method (as in R)
            if len(p_values) > 1:
                # Fisher's method: -2 * sum(log(p))
                # Avoid log(0) by using small value
                p_values = np.clip(p_values, 1e-300, 1)
                fisher_stat = -2 * np.sum(np.log(p_values))

                # Combined p-value from chi-squared distribution
                # Degrees of freedom = 2 * number of p-values
                combined_p = 1 - stats.chi2.cdf(fisher_stat, 2 * len(p_values))
            else:
                combined_p = p_values[0] if p_values else 1.0

            # Test against FDR threshold
            if combined_p < FDR:
                # Evidence of expression - exclude cluster
                cluster_passes = False
                break

        if not cluster_passes:
            use_cells[cell_indices] = False

    if verbose:
        n_excluded = np.sum(~use_cells)
        n_excluded_clusters = len(unique_clusters) - len(np.unique(clusters[use_cells]))
        print(f"Excluded {n_excluded} cells in {n_excluded_clusters} clusters")

    return use_cells

def quickMarkers(
        toc: sparse.csr_matrix,
        clusters: np.ndarray,
        N: Optional[int] = 10,
        FDR: float = 0.01,
        expressCut = 0.9,
        verbose: bool = True,
        gene_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Find marker genes using tf-idf method.
    Matches R's quickMarkers implementation.

    Parameters
    ----------
    toc : sparse matrix
        Count matrix (genes x cells)
    clusters : array
        Cluster assignments
    N : int or None
        Number of markers per cluster. None = all markers (Inf in R)
    FDR : float
        False discovery rate threshold
    verbose : bool
        Print progress
    gene_names : list, optional
        List of gene names. If None, uses indices

    Returns
    -------
    pd.DataFrame
        Marker genes with columns: gene, cluster, tfidf, geneFrequency
    """
    unique_clusters = np.unique(clusters)
    n_genes = toc.shape[0]

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    if verbose:
        print(f"Finding markers for {len(unique_clusters)} clusters")

    markers = []

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        n_cells_cluster = np.sum(cluster_mask)
        n_cells_total = len(clusters)

        if n_cells_cluster == 0:
            continue

        # Calculate gene frequencies
        cluster_counts = toc[:, cluster_mask]
        global_counts = toc

        # Binarize counts to match R (use expressCut=0.9 if needed, else >0)
        expressCut = expressCut  # Match R's default (expressCut=0.9 if specified)
        cells_expressing_cluster = np.array((cluster_counts > expressCut).sum(axis=1)).flatten()
        cells_expressing_global = np.array((global_counts > expressCut).sum(axis=1)).flatten()

        # Gene frequencies
        freq_cluster = cells_expressing_cluster / n_cells_cluster
        freq_global = cells_expressing_global / n_cells_total

        # Calculate tf-idf scores
        freq_global_safe = np.maximum(freq_global, 1e-10)
        idf = np.log(n_cells_total / freq_global_safe)
        tfidf_scores = freq_cluster * idf

        # Statistical test for enrichment (hypergeometric)
        p_values = []
        for gene_idx in range(n_genes):
            M = n_cells_total
            n = cells_expressing_global[gene_idx]
            N_sample = n_cells_cluster
            k = cells_expressing_cluster[gene_idx]

            if n == 0 or k == 0:
                p_values.append(1.0)
            else:
                # One-sided test: over-representation
                p_val = stats.hypergeom.sf(k - 1, M, n, N_sample)
                p_values.append(p_val)

        # FDR correction
        if len(p_values) > 0:
            _, p_adjusted, _, _ = multipletests(p_values, alpha=FDR, method='fdr_bh')
        else:
            p_adjusted = p_values

        # Select significant markers
        significant = np.array(p_adjusted) < FDR

        # Sort by tf-idf score
        marker_indices = np.where(significant)[0]
        marker_indices = marker_indices[np.argsort(-tfidf_scores[marker_indices])]

        # Limit to top N if specified
        if N is not None and len(marker_indices) > N:
            marker_indices = marker_indices[:N]

        # Store markers with gene names
        for gene_idx in marker_indices:
            markers.append({
                'gene': gene_names[gene_idx],
                'cluster': cluster_id,
                'tfidf': tfidf_scores[gene_idx],
                'geneFrequency': freq_cluster[gene_idx],
                'geneFrequencyGlobal': freq_global[gene_idx],
                'p_value': p_values[gene_idx],
                'p_adjusted': p_adjusted[gene_idx]
            })

    df = pd.DataFrame(markers)

    if verbose and len(df) > 0:
        print(f"Found {len(df)} markers total")

    return df


# Backwards compatibility functions
def estimate_soup(sc: "SoupChannel") -> pd.DataFrame:
    """
    Estimate soup profile from empty droplets.
    For backwards compatibility.
    """
    sc._calculate_soup_profile()

    # Normalize soup profile to match R (if not already normalized)
    sc.soupProfile['est'] = sc.soupProfile['counts'] / sc.soupProfile['counts'].sum()

    return sc.soupProfile


def calculate_contamination_fraction(
        sc: "SoupChannel",
        non_expressed_genes: list,
        clusters: Optional[np.ndarray] = None
) -> float:
    """
    Simple contamination estimation for backwards compatibility.
    Use autoEstCont for better results.
    """
    if clusters is None:
        clusters = sc.metaData.get('clusters', np.zeros(sc.n_cells))

    # Get cells not expressing these genes
    non_expressing = estimateNonExpressingCells(
        sc, non_expressed_genes, clusters,
        maximumContamination=0.5,
        FDR=0.1,
        verbose=False
    )

    # Calculate contamination from non-expressing cells
    contamination_estimates = []

    for gene in non_expressed_genes:
        if gene not in sc.soupProfile.index:
            continue

        gene_idx = sc.soupProfile.index.get_loc(gene)
        soup_expression = sc.soupProfile.iloc[gene_idx]['est']

        # Use only non-expressing cells
        non_expr_indices = np.where(non_expressing)[0]
        if len(non_expr_indices) == 0:
            continue

        for cell_idx in non_expr_indices:
            observed = sc.toc[gene_idx, cell_idx]
            expected_soup = soup_expression * sc.metaData.iloc[cell_idx]['nUMIs']

            if expected_soup > 0:
                rho = observed / expected_soup
                contamination_estimates.append(min(rho, 1.0))

    if len(contamination_estimates) == 0:
        print("Warning: Could not estimate contamination. Using default 0.1")
        return 0.1

    # Return median estimate
    return np.median(contamination_estimates)


# Wrapper for old function names
def auto_est_cont(*args, **kwargs):
    """Backwards compatible wrapper."""
    return autoEstCont(*args, **kwargs)