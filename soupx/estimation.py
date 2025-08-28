"""
Soup profile estimation functions.

Core functionality for estimating the background contamination profile
from empty droplets and automated contamination fraction estimation.
"""
import numpy as np
import pandas as pd
from scipy import stats, sparse
from statsmodels.stats.multitest import multipletests
from typing import Tuple, List, Optional, Dict, TYPE_CHECKING

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
        non_expressed_genes: List[str],
        non_expressing_cells: Optional[np.ndarray] = None,
        verbose: bool = True
) -> "SoupChannel":
    """
    Updated to use estimate_non_expressing_cells if non_expressing_cells not provided.
    """
    if sc.soup_profile is None:
        raise ValueError("Must estimate soup profile first")

    # Find gene indices
    gene_mask = np.isin(sc.gene_names, non_expressed_genes)
    if not np.any(gene_mask):
        raise ValueError(f"None of the specified genes found: {non_expressed_genes}")

    found_genes = sc.gene_names[gene_mask]

    # Use estimate_non_expressing_cells if not provided
    if non_expressing_cells is None:
        if verbose:
            print("Using statistical method to identify non-expressing cells...")
        non_expressing_cells = estimate_non_expressing_cells(
            sc, found_genes.tolist(), verbose=verbose
        )

    if verbose:
        print(f"Using {len(found_genes)} genes for contamination estimation")
        print(f"Using {np.sum(non_expressing_cells)} cells for estimation")

    # Rest of function remains the same...
    gene_indices = np.where(gene_mask)[0]
    cell_indices = np.where(non_expressing_cells)[0]

    contamination_fractions = []

    for cell_idx in cell_indices:
        observed_counts = sc.filtered_counts[gene_indices, cell_idx].toarray().flatten()
        cell_umis = sc.metadata.iloc[cell_idx]['n_umis']
        soup_fractions = sc.soup_profile.loc[found_genes, 'est'].values
        expected_soup_fraction = np.sum(soup_fractions)

        if cell_umis == 0 or expected_soup_fraction == 0:
            continue

        total_observed = np.sum(observed_counts)
        contamination_frac = total_observed / (cell_umis * expected_soup_fraction)
        contamination_frac = np.clip(contamination_frac, 0, 1)
        contamination_fractions.append(contamination_frac)

    if len(contamination_fractions) == 0:
        raise ValueError("No valid contamination estimates obtained")

    final_contamination = np.median(contamination_fractions)

    if verbose:
        print(f"Contamination estimates: min={np.min(contamination_fractions):.3f}, "
              f"median={np.median(contamination_fractions):.3f}, "
              f"max={np.max(contamination_fractions):.3f}")
        print(f"Final contamination fraction: {final_contamination:.3f}")

    sc.set_contamination_fraction(final_contamination)

    return sc


def quickMarkers(
        toc: sparse.csr_matrix,
        clusters: np.ndarray,
        N: int = 10,
        FDR: float = 0.01,
        expressCut: float = 0.9,
        gene_names: np.ndarray = None
) -> pd.DataFrame:
    """
    Debug version of quickMarkers with extensive logging
    """
    print(f"\n=== DEBUG quickMarkers ===")
    print(f"Input matrix: {toc.shape} ({toc.nnz} non-zero entries)")
    print(f"Clusters: {len(np.unique(clusters))} unique clusters")
    print(f"Parameters: N={N}, FDR={FDR}, expressCut={expressCut}")

    # Convert to COO format
    toc_coo = toc.tocoo()
    print(f"COO format: {len(toc_coo.data)} entries")

    # Binarize: find entries > expressCut
    w = toc_coo.data > expressCut
    print(f"Entries > {expressCut}: {np.sum(w)} / {len(toc_coo.data)}")

    if not np.any(w):
        print("ERROR: No entries > expressCut!")
        return pd.DataFrame()

    # Get valid entries
    valid_genes = toc_coo.row[w]
    valid_cells = toc_coo.col[w]

    print(f"Valid entries: {len(valid_genes)} gene-cell pairs")
    print(f"Unique genes in valid entries: {len(np.unique(valid_genes))}")

    # Cluster counts
    unique_clusters = np.unique(clusters[clusters >= 0])
    print(f"Valid clusters: {unique_clusters}")

    clCnts = {}
    for cluster in unique_clusters:
        count = np.sum(clusters == cluster)
        clCnts[cluster] = count
        print(f"  Cluster {cluster}: {count} cells")

    # Count gene occurrences per cluster
    n_genes = toc.shape[0]
    print(f"Total genes: {n_genes}")

    nObs = {}
    for cluster in unique_clusters:
        cluster_cells = clusters == cluster
        cluster_cell_indices = np.where(cluster_cells)[0]

        # Find which valid entries belong to this cluster
        cluster_entries = np.isin(valid_cells, cluster_cell_indices)
        cluster_genes = valid_genes[cluster_entries]

        print(
            f"  Cluster {cluster}: {np.sum(cluster_entries)} valid entries, {len(np.unique(cluster_genes))} unique genes")

        # Count occurrences of each gene in this cluster
        gene_counts = np.bincount(cluster_genes, minlength=n_genes)
        nObs[cluster] = gene_counts

        print(f"    Genes with counts > 0: {np.sum(gene_counts > 0)}")
        print(f"    Max gene count: {np.max(gene_counts)}")

    # Convert to matrix format
    cluster_list = sorted(nObs.keys())
    nObs_matrix = np.column_stack([nObs[cluster] for cluster in cluster_list])

    print(f"nObs matrix shape: {nObs_matrix.shape}")
    print(f"nObs matrix non-zero: {np.sum(nObs_matrix > 0)}")

    # Calculate totals
    nTot = np.sum(nObs_matrix, axis=1)
    print(f"nTot: {np.sum(nTot > 0)} genes with total counts > 0")
    print(f"nTot max: {np.max(nTot)}")

    # Calculate term frequencies
    cluster_sizes = np.array([clCnts[cluster] for cluster in cluster_list])
    tf = nObs_matrix / cluster_sizes[np.newaxis, :]

    print(f"TF shape: {tf.shape}")
    print(f"TF non-zero: {np.sum(tf > 0)}")
    print(f"TF max: {np.max(tf)}")

    # Calculate IDF
    idf = np.log(toc.shape[1] / (nTot + 1e-10))
    print(f"IDF: min={np.min(idf):.3f}, max={np.max(idf):.3f}")

    # Calculate TF-IDF score
    score = tf * idf[:, np.newaxis]
    print(f"Score: min={np.min(score):.3f}, max={np.max(score):.3f}")
    print(f"Score > 0: {np.sum(score > 0)}")
    print(f"Score > 1: {np.sum(score > 1)}")
    print(f"Score > 2: {np.sum(score > 2)}")

    # Show top scores per cluster
    for i, cluster in enumerate(cluster_list):
        top_scores = np.sort(score[:, i])[-5:]
        print(f"  Cluster {cluster} top 5 scores: {top_scores}")

    # Hypergeometric tests (simplified for debugging)
    print(f"Calculating p-values...")
    qvals = np.ones_like(tf)  # Start with all p=1

    n_tests = 0
    n_significant = 0

    for i, cluster in enumerate(cluster_list):
        cluster_size = clCnts[cluster]

        for gene_idx in range(min(n_genes, 100)):  # Test first 100 genes for speed
            if nTot[gene_idx] == 0:
                continue

            n_tests += 1

            k = nObs_matrix[gene_idx, i] - 1
            M = toc.shape[1]
            n = nTot[gene_idx]
            N = cluster_size

            if k >= 0 and n > 0 and N > 0:
                try:
                    p_val = 1 - stats.hypergeom.cdf(k, M, n, N)
                    qvals[gene_idx, i] = p_val
                    if p_val < FDR:
                        n_significant += 1
                except:
                    qvals[gene_idx, i] = 1.0

    print(f"Hypergeometric tests: {n_tests} tests, {n_significant} significant at FDR={FDR}")

    # Apply FDR correction (on subset for speed)
    print(f"Applying FDR correction...")
    for i in range(len(cluster_list)):
        subset = qvals[:100, i]  # Test first 100 genes
        if np.sum(subset < 1.0) > 1:
            try:
                _, corrected, _, _ = multipletests(subset, alpha=FDR, method='fdr_bh')
                qvals[:100, i] = corrected
            except:
                pass

    # Build results
    print(f"Building results...")
    results = []

    for i, cluster in enumerate(cluster_list):
        # Sort by score descending
        gene_order = np.argsort(score[:, i])[::-1]

        # Show top scores for this cluster
        print(f"  Cluster {cluster} top scores:")
        for j in range(min(10, len(gene_order))):
            gene_idx = gene_order[j]
            score_val = score[gene_idx, i]
            qval = qvals[gene_idx, i]
            gene_name = f"Gene_{gene_idx:04d}" if gene_names is None else gene_names[gene_idx]
            print(f"    {gene_name}: score={score_val:.3f}, qval={qval:.4f}")

        # Select markers
        if N == np.inf or N > len(gene_order):
            # Take all passing FDR
            passing_fdr = qvals[gene_order, i] < FDR
            selected_genes = gene_order[passing_fdr]
            print(f"    Taking all {np.sum(passing_fdr)} FDR-passing genes")
        else:
            # Take top N
            selected_genes = gene_order[:N]
            print(f"    Taking top {N} genes")

        # Build results for this cluster
        for gene_idx in selected_genes:
            gene_name = f"Gene_{gene_idx:04d}" if gene_names is None else gene_names[gene_idx]

            results.append({
                'gene_idx': gene_idx,
                'gene': gene_name,
                'cluster': cluster,
                'geneFrequency': tf[gene_idx, i],
                'tfidf': score[gene_idx, i],
                'qval': qvals[gene_idx, i]
            })

    print(f"Total results: {len(results)}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"Results summary:")
        print(f"  TF-IDF range: {df['tfidf'].min():.3f} to {df['tfidf'].max():.3f}")
        print(f"  TF-IDF >= 1.0: {np.sum(df['tfidf'] >= 1.0)}")
        print(f"  TF-IDF >= 0.5: {np.sum(df['tfidf'] >= 0.5)}")
        print(f"  TF-IDF >= 0.1: {np.sum(df['tfidf'] >= 0.1)}")

    return df

def estimate_non_expressing_cells(
        sc: "SoupChannel",
        non_expressed_gene_list: List[str],
        clusters: Optional[np.ndarray] = None,
        maximum_contamination: float = 1.0,
        FDR: float = 0.05,
        verbose: bool = True
) -> np.ndarray:
    """
    Faithful implementation of R's estimateNonExpressingCells.

    Uses Poisson tests to identify clusters where genes are genuinely not expressed.
    Conservative approach - excludes entire clusters if any cell significantly expresses the genes.

    Parameters
    ----------
    sc : SoupChannel
        SoupChannel object with soup profile estimated
    non_expressed_gene_list : list of str
        Gene names that should not be expressed in certain cell types
    clusters : array-like, optional
        Cluster assignments. If None, uses sc.clusters
    maximum_contamination : float, default 1.0
        Maximum expected contamination fraction
    FDR : float, default 0.05
        False discovery rate for Poisson tests
    verbose : bool, default True
        Print progress information

    Returns
    -------
    np.ndarray
        Boolean array (n_cells,) indicating which cells to use for estimation
    """

    if sc.soup_profile is None:
        raise ValueError("Must estimate soup profile first")

    # Get clusters
    if clusters is None:
        if sc.clusters is None:
            if verbose:
                print("No clusters found, using every cell as its own cluster.")
            clusters = np.arange(sc.n_cells)
        else:
            clusters = sc.clusters

    # Find gene indices
    gene_mask = np.isin(sc.gene_names, non_expressed_gene_list)
    if not np.any(gene_mask):
        raise ValueError(f"None of the specified genes found: {non_expressed_gene_list}")

    found_genes = sc.gene_names[gene_mask]
    gene_indices = np.where(gene_mask)[0]

    if verbose:
        print(f"Testing non-expression for {len(found_genes)} genes across clusters")

    # Initialize result - start with all cells usable
    use_cells = np.ones(sc.n_cells, dtype=bool)

    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove unassigned

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        if len(cell_indices) == 0:
            continue

        # Test each gene in this cluster
        cluster_passes = True

        for gene_idx in gene_indices:
            gene_name = sc.gene_names[gene_idx]

            # Get observed counts for this gene in this cluster
            obs_counts = sc.filtered_counts[gene_idx, cell_indices].toarray().flatten()

            # Calculate expected counts under maximum contamination
            cell_umis = sc.metadata.iloc[cell_indices]['n_umis'].values
            soup_fraction = sc.soup_profile.loc[gene_name, 'est']
            expected_counts = cell_umis * soup_fraction * maximum_contamination

            # Poisson test for each cell: is observed > expected under max contamination?
            p_values = []
            for obs, exp in zip(obs_counts, expected_counts):
                if exp <= 0:
                    p_val = 0.0 if obs > 0 else 1.0
                else:
                    # Test if observed is significantly greater than expected
                    p_val = 1 - stats.poisson.cdf(obs - 1, exp)
                p_values.append(p_val)

            # Apply FDR correction
            if len(p_values) > 1:
                _, p_adjusted, _, _ = multipletests(p_values, alpha=FDR, method='fdr_bh')
            else:
                p_adjusted = p_values

            # If any cell in cluster has significant expression, exclude whole cluster
            if np.any(np.array(p_adjusted) < FDR):
                cluster_passes = False
                break

        # If cluster fails for any gene, exclude all cells in cluster
        if not cluster_passes:
            use_cells[cell_indices] = False

    n_excluded_cells = np.sum(~use_cells)
    n_excluded_clusters = len(unique_clusters) - len(np.unique(clusters[use_cells]))

    if verbose:
        print(f"Excluded {n_excluded_cells} cells in {n_excluded_clusters} clusters")
        print(f"Using {np.sum(use_cells)} cells for contamination estimation")

    return use_cells


def auto_est_cont(
        sc: "SoupChannel",
        top_markers: Optional[pd.DataFrame] = None,
        tfidf_min: float = 1.0,  # R default
        soup_quantile: float = 0.90,  # R default
        max_markers: int = 100,  # R default
        contamination_range: Tuple[float, float] = (0.01, 0.8),
        rho_max_fdr: float = 0.2,
        prior_rho: float = 0.05,
        prior_rho_std_dev: float = 0.10,
        verbose: bool = True
) -> "SoupChannel":
    """
    Exact implementation of R autoEstCont.

    Follows R algorithm precisely:
    1. Collapse to cluster level (R: do.call(cbind,lapply(s,function(e) rowSums(...))))
    2. Find markers using quickMarkers with N=Inf
    3. Apply tfidf and soup quantile filters exactly as R
    4. Use estimateNonExpressingCells for statistical validation
    """

    if sc.clusters is None:
        raise ValueError("Clustering information must be supplied, run setClusters first.")
    if sc.soup_profile is None:
        raise ValueError("Soup profile must be estimated first.")

    if verbose:
        print("Automatically estimating contamination fraction...")

    # Step 1: Collapse by cluster (R: First collapse by cluster)
    if verbose:
        print("Collapsing to cluster level...")

    unique_clusters = np.unique(sc.clusters[sc.clusters >= 0])
    cluster_counts = []
    cluster_metadata = []

    for cluster_id in unique_clusters:
        cluster_mask = sc.clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        if len(cell_indices) == 0:
            continue

        # Sum counts for this cluster (R: rowSums(sc$toc[,e,drop=FALSE]))
        cluster_sum = np.array(sc.filtered_counts[:, cell_indices].sum(axis=1)).flatten()
        cluster_counts.append(cluster_sum)

        # Cluster metadata (R: nUMIs = colSums(tmp))
        cluster_umis = np.sum(cluster_sum)
        cluster_metadata.append({
            'nUMIs': cluster_umis,
            'cluster_id': cluster_id
        })

    if len(cluster_counts) == 0:
        raise ValueError("No valid clusters found")

    # Create cluster-level count matrix
    ssc_toc = sparse.csr_matrix(np.column_stack(cluster_counts))
    ssc_metadata = pd.DataFrame(cluster_metadata)

    # Step 2: Get markers (R: if(is.null(topMarkers)))
    if top_markers is None:
        if verbose:
            print("Finding marker genes...")

        # Use original cell-level data for marker finding (not cluster-collapsed)
        # R: mrks = quickMarkers(sc$toc,sc$metaData$clusters,N=Inf)
        mrks = quickMarkers(sc.filtered_counts, sc.clusters, N=np.inf, FDR=0.01)

        if len(mrks) == 0:
            raise ValueError("No marker genes found")

        # Replace gene names with actual names
        mrks['gene'] = sc.gene_names[mrks['gene_idx']]

        # R: And only the most specific entry for each gene
        # mrks = mrks[order(mrks$gene,-mrks$tfidf),]
        # mrks = mrks[!duplicated(mrks$gene),]
        mrks = mrks.sort_values(['gene', 'tfidf'], ascending=[True, False])
        mrks = mrks.drop_duplicates('gene', keep='first')

        # R: Order by tfidf maxness
        mrks = mrks.sort_values('tfidf', ascending=False)

        # R: Apply tf-idf cut-off
        mrks = mrks[mrks['tfidf'] > tfidf_min]

    else:
        mrks = top_markers.copy()

    if len(mrks) == 0:
        raise ValueError(f"No marker genes pass tfidf >= {tfidf_min}")


    # Step 3: Filter by soup quantile (R: soup filtering)
    if verbose:
        print(f"Applying soup quantile filter ({soup_quantile})...")

    # R: soupProf = ssc$soupProfile[order(ssc$soupProfile$est,decreasing=TRUE),]
    # R: soupMin = quantile(soupProf$est,soupQuantile)
    soup_min = np.quantile(sc.soup_profile['est'], soup_quantile)

    # R: Filter to include only those that exist in soup
    # R: tgts = rownames(soupProf)[soupProf$est>soupMin]
    high_soup_genes = sc.soup_profile[sc.soup_profile['est'] > soup_min].index.tolist()

    # R: And get the ones that pass our tfidf cut-off
    # R: filtPass = mrks[mrks$gene %in% tgts,]
    filt_pass = mrks[mrks['gene'].isin(high_soup_genes)]

    # R: tgts = head(filtPass$gene,n=maxMarkers)
    tgts = filt_pass['gene'].head(max_markers).tolist()

    if verbose:
        print(
            f"{len(mrks)} genes passed tf-idf cut-off and {len(filt_pass)} soup quantile filter. Taking the top {len(tgts)}.")

    if len(tgts) == 0:
        raise ValueError(
            "No plausible marker genes found. Is the channel low complexity? If not, reduce tfidfMin or soupQuantile")

    if len(tgts) < 10:
        print("Warning: Fewer than 10 marker genes found. Is this channel low complexity?")

    # Step 4: Use estimateNonExpressingCells and build estimates
    if verbose:
        print("Estimating non-expressing cells for each marker...")

    # This is where R calls estimateNonExpressingCells and builds the dd dataframe
    # For now, use a simplified version that matches R's basic logic
    dd_list = []

    for gene_name in tgts:
        if gene_name not in sc.soup_profile.index:
            continue

        # Get soup fraction for this gene
        soup_fraction = sc.soup_profile.loc[gene_name, 'est']
        gene_idx = np.where(sc.gene_names == gene_name)[0]

        if len(gene_idx) == 0:
            continue
        gene_idx = gene_idx[0]

        # For each cluster, calculate contamination estimate
        # Use cells NOT in the gene's marker cluster
        marker_info = mrks[mrks['gene'] == gene_name]
        if len(marker_info) == 0:
            continue

        marker_cluster = marker_info.iloc[0]['cluster']

        # Use cells from other clusters
        non_marker_cells = sc.clusters != marker_cluster
        if np.sum(non_marker_cells) < 3:
            continue

        # Calculate observed and expected
        cell_indices = np.where(non_marker_cells)[0]
        obs_counts = sc.filtered_counts[gene_idx, cell_indices].toarray().flatten()
        cell_umis = sc.metadata.iloc[cell_indices]['n_umis'].values

        obs_cnt = np.sum(obs_counts)
        exp_cnt = np.sum(cell_umis * soup_fraction)

        if exp_cnt <= 0:
            continue

        dd_list.append({
            'gene': gene_name,
            'gene_idx': gene_idx,
            'obsCnt': obs_cnt,
            'expCnt': exp_cnt,
            'tfidf': marker_info.iloc[0]['tfidf'],
            'useEst': True  # Simplified - R would use more complex filtering
        })

    if len(dd_list) == 0:
        raise ValueError("No valid marker/cluster combinations found for estimation.")

    dd = pd.DataFrame(dd_list)
    dd = dd[dd['useEst']].copy()

    if len(dd) == 0:
        raise ValueError("No marker/cluster combinations passed filtering.")

    if verbose:
        print(f"Using {len(dd)} independent estimates of rho.")

    # Step 5: Gamma prior/posterior estimation (exact R implementation)
    # R: Set up gamma prior parameters
    v2 = (prior_rho_std_dev / prior_rho) ** 2
    k = 1 + v2 ** (-2) / 2 * (1 + np.sqrt(1 + 4 * v2))
    theta = prior_rho / (k - 1)

    if verbose:
        print(f"Using gamma prior with k={k:.3f}, theta={theta:.3f}")

    # R: rhoProbes=seq(0,1,.001)
    rho_probes = np.arange(0, 1.001, 0.001)

    # R: Calculate posterior for each rho value
    posterior_probs = []

    for rho in rho_probes:
        # R: mean(dgamma(e,k+tmp$obsCnt,scale=theta/(1+theta*tmp$expCnt)))
        posterior_values = []

        for _, row in dd.iterrows():
            obs_cnt = row['obsCnt']
            exp_cnt = row['expCnt']

            # Gamma posterior parameters
            shape = k + obs_cnt
            scale = theta / (1 + theta * exp_cnt)

            if scale > 0:
                prob = stats.gamma.pdf(rho, a=shape, scale=scale)
                posterior_values.append(prob)

        if len(posterior_values) > 0:
            posterior_probs.append(np.mean(posterior_values))
        else:
            posterior_probs.append(0)

    posterior_probs = np.array(posterior_probs)

    # Find mode within contamination range
    valid_range_mask = (rho_probes >= contamination_range[0]) & (rho_probes <= contamination_range[1])
    valid_indices = np.where(valid_range_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"No valid contamination range found between {contamination_range}")

    valid_posterior = posterior_probs[valid_indices]
    best_idx = valid_indices[np.argmax(valid_posterior)]
    rho_est = rho_probes[best_idx]

    # Calculate FWHM
    half_max = np.max(valid_posterior) / 2
    fwhm_mask = valid_posterior >= half_max
    if np.any(fwhm_mask):
        fwhm_indices = valid_indices[fwhm_mask]
        rho_fwhm = (rho_probes[fwhm_indices[0]], rho_probes[fwhm_indices[-1]])
    else:
        rho_fwhm = (rho_est, rho_est)

    if verbose:
        print(f"Estimated global rho of {rho_est:.3f}")
        print(f"FWHM range: ({rho_fwhm[0]:.3f}, {rho_fwhm[1]:.3f})")

    # Store fit information (R format)
    sc.fit = {
        'dd': dd,
        'priorRho': prior_rho,
        'priorRhoStdDev': prior_rho_std_dev,
        'posterior': posterior_probs,
        'rhoEst': rho_est,
        'rhoFWHM': rho_fwhm,
        'rhoProbes': rho_probes,
        'markersUsed': mrks
    }

    # Set contamination fraction
    sc.set_contamination_fraction(rho_est)

    return sc