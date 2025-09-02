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


def autoEstCont(sc, **kwargs):
    """
    Exact R implementation matching cluster-level aggregation and gene×cluster matrix approach.
    """
    from scipy import stats
    import numpy as np
    import pandas as pd

    # Extract parameters with R defaults
    tfidfMin = kwargs.get('tfidfMin', 1.0)
    soupQuantile = kwargs.get('soupQuantile', 0.90)
    maxMarkers = kwargs.get('maxMarkers', 100)
    contaminationRange = kwargs.get('contaminationRange', (0.01, 0.8))
    rhoMaxFDR = kwargs.get('rhoMaxFDR', 0.2)
    priorRho = kwargs.get('priorRho', 0.05)
    priorRhoStdDev = kwargs.get('priorRhoStdDev', 0.10)
    verbose = kwargs.get('verbose', True)
    forceAccept = kwargs.get('forceAccept', False)

    if 'clusters' not in sc.metaData.columns:
        raise ValueError("Clustering information must be supplied, run setClusters first.")

    clusters = sc.metaData['clusters'].values
    unique_clusters = np.unique(clusters)

    # STEP 1: Collapse by cluster - EXACT R MATCH
    # R: s = split(rownames(sc$metaData),sc$metaData$clusters)
    # R: tmp = do.call(cbind,lapply(s,function(e) rowSums(sc$toc[,e,drop=FALSE])))
    cluster_toc = []
    cluster_metadata = []

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cell_indices = np.where(cluster_mask)[0]

        # R: rowSums(sc$toc[,e,drop=FALSE])
        cluster_counts = np.array(sc.toc[:, cell_indices].sum(axis=1)).flatten()
        cluster_toc.append(cluster_counts)

        # R: data.frame(nUMIs = colSums(tmp))
        cluster_nUMIs = cluster_counts.sum()
        cluster_metadata.append({'nUMIs': cluster_nUMIs})

    # Create cluster-level data structures
    cluster_toc_matrix = np.column_stack(cluster_toc)  # genes × clusters
    cluster_metaData = pd.DataFrame(cluster_metadata, index=unique_clusters)

    if verbose:
        print(f"Collapsed to cluster level: {cluster_toc_matrix.shape} matrix")

    # STEP 2: Get markers - R does this on original cell-level data
    # R: mrks = quickMarkers(sc$toc,sc$metaData$clusters,N=Inf)
    mrks = quickMarkers(
        sc.toc, clusters, N=None, verbose=False,
        gene_names=sc.gene_names if hasattr(sc, 'gene_names') else None,
        expressCut=0.9
    )

    # R marker processing
    mrks = mrks.sort_values(['gene', 'tfidf'], ascending=[True, False])
    mrks = mrks[~mrks.duplicated(subset='gene', keep='first')]
    mrks = mrks.sort_values('tfidf', ascending=False)
    mrks = mrks[mrks['tfidf'] > tfidfMin]

    # STEP 3: Filter by soup quantile
    # R: soupProf = ssc$soupProfile[order(ssc$soupProfile$est,decreasing=TRUE),]
    # R: soupMin = quantile(soupProf$est,soupQuantile)
    soupProf = sc.soupProfile.sort_values('est', ascending=False)
    soupMin = np.quantile(soupProf['est'].values, soupQuantile)

    # R: tgts = rownames(soupProf)[soupProf$est>soupMin]
    tgts = soupProf.index[soupProf['est'] > soupMin].tolist()

    # R: filtPass = mrks[mrks$gene %in% tgts,]
    # R: tgts = head(filtPass$gene,n=maxMarkers)
    filtPass = mrks[mrks['gene'].isin(tgts)]
    final_genes = filtPass.head(maxMarkers)['gene'].tolist()

    if verbose:
        print(f"{len(mrks)} genes passed tf-idf cut-off and {len(filtPass)} soup quantile filter. "
              f"Taking the top {len(final_genes)}.")

    if len(final_genes) == 0:
        raise ValueError("No plausible marker genes found. Reduce tfidfMin or soupQuantile")

    if len(final_genes) < 10:
        print("Warning: Fewer than 10 marker genes found. Is this channel low complexity?")

    # STEP 4: Get estimates in clusters - CORRECTED R METHOD
    # R: tmp = as.list(tgts); names(tmp) = tgts
    # R creates a list where each gene is its own "gene set"
    gene_sets_dict = {gene: [gene] for gene in final_genes}

    # R: ute = estimateNonExpressingCells(sc,tmp,maximumContamination=max(contaminationRange),FDR=rhoMaxFDR)
    # Call estimateNonExpressingCells ONCE with all genes as individual gene sets
    ute_cell_level = np.zeros((sc.n_cells, len(final_genes)), dtype=bool)

    # This is the critical fix - call estimateNonExpressingCells for each gene as R does
    for i, gene in enumerate(final_genes):
        non_expressing = estimateNonExpressingCells(
            sc, [gene], clusters,  # Single gene as list (like R's individual gene sets)
            maximumContamination=max(contaminationRange),
            FDR=rhoMaxFDR,
            verbose=False
        )
        ute_cell_level[:, i] = non_expressing

    # R: m = rownames(sc$metaData)[match(rownames(ssc$metaData),sc$metaData$clusters)]
    # R: ute = t(ute[m,,drop=FALSE])
    # Transform from cell-level to cluster-level matrix
    ute = np.zeros((len(final_genes), len(unique_clusters)), dtype=bool)

    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        # For each gene, check if ANY cell in this cluster can be used for estimation
        for j in range(len(final_genes)):
            ute[j, i] = np.any(ute_cell_level[cluster_mask, j])

    if verbose:
        n_usable_combinations = np.sum(ute)
        print(f"Found {n_usable_combinations} usable gene×cluster combinations for estimation")

    # STEP 5: Create gene×cluster matrices - EXACT R
    # R: expCnts = outer(ssc$soupProfile$est,ssc$metaData$nUMIs)
    soup_est = sc.soupProfile.loc[final_genes, 'est'].values
    cluster_nUMIs = cluster_metaData['nUMIs'].values
    expCnts = np.outer(soup_est, cluster_nUMIs)  # genes × clusters

    # R: obsCnts = ssc$toc[tgts,,drop=FALSE]
    gene_indices = [sc.soupProfile.index.get_loc(gene) for gene in final_genes]
    obsCnts = cluster_toc_matrix[gene_indices, :]  # genes × clusters

    # STEP 6: Create data frame with all gene×cluster combinations
    # R: dd = data.frame(gene = rep(rownames(ute),ncol(ute)), ...)
    n_genes = len(final_genes)
    n_clusters = len(unique_clusters)

    # Create vectors for all gene×cluster combinations
    genes_vec = np.repeat(final_genes, n_clusters)
    clusters_vec = np.tile(unique_clusters, n_genes)
    passNonExp_vec = ute.flatten()
    rhoEst_vec = (obsCnts / expCnts).flatten()
    obsCnt_vec = obsCnts.flatten()
    expCnt_vec = expCnts.flatten()

    # Create the data frame
    dd = pd.DataFrame({
        'gene': genes_vec,
        'cluster': clusters_vec,
        'passNonExp': passNonExp_vec,
        'rhoEst': rhoEst_vec,
        'obsCnt': obsCnt_vec,
        'expCnt': expCnt_vec
    })

    # R: dd$useEst = dd$passNonExp
    dd['useEst'] = dd['passNonExp']

    n_estimates = dd['useEst'].sum()
    if n_estimates < 10:
        print("Warning: Fewer than 10 independent estimates, rho estimation may be unstable.")

    if verbose:
        print(f"Using {n_estimates} independent estimates of rho.")

    # STEP 7: Aggregate posterior probabilities - EXACT R
    # R: v2 = (priorRhoStdDev/priorRho)**2
    # R: k = 1 +v2**-2/2*(1+sqrt(1+4*v2))
    # R: theta = priorRho/(k-1)
    v2 = (priorRhoStdDev / priorRho) ** 2
    k = 1 + v2 ** -2 / 2 * (1 + np.sqrt(1 + 4 * v2))
    theta = priorRho / (k - 1)

    # R: rhoProbes=seq(0,1,.001)
    rhoProbes = np.arange(0, 1.001, 0.001)

    # R: tmp = sapply(rhoProbes,function(e) {
    #        tmp = dd[dd$useEst,]
    #        mean(dgamma(e,k+tmp$obsCnt,scale=theta/(1+theta*tmp$expCnt)))
    #     })
    valid_estimates = dd[dd['useEst']]

    posterior_density = []
    for rho in rhoProbes:
        densities = []
        for _, row in valid_estimates.iterrows():
            # R: dgamma(e,k+tmp$obsCnt,scale=theta/(1+theta*tmp$expCnt))
            shape = k + row['obsCnt']
            scale = theta / (1 + theta * row['expCnt'])
            density = stats.gamma.pdf(rho, a=shape, scale=scale)
            densities.append(density)

        posterior_density.append(np.mean(densities) if densities else 0)

    posterior_density = np.array(posterior_density)

    # R: w = which(rhoProbes>=contaminationRange[1] & rhoProbes<=contaminationRange[2])
    # R: rhoEst = (rhoProbes[w])[which.max(tmp[w])]
    valid_range = (rhoProbes >= contaminationRange[0]) & (rhoProbes <= contaminationRange[1])
    valid_idx = np.where(valid_range)[0]

    if len(valid_idx) == 0:
        raise ValueError("No valid rho values in contamination range")

    peak_idx = valid_idx[np.argmax(posterior_density[valid_idx])]
    rhoEst = rhoProbes[peak_idx]

    if verbose:
        print(f"Estimated global rho of {rhoEst:.2f}")

    # Set contamination fraction
    sc.set_contamination_fraction(rhoEst, forceAccept=forceAccept)

    # Store fit information
    sc.fit = {
        'dd': dd,
        'estimates': dd[dd['useEst']].copy(),
        'priorRho': priorRho,
        'priorRhoStdDev': priorRhoStdDev,
        'posterior': posterior_density,
        'rhoEst': rhoEst,
        'rhoProbes': rhoProbes,
        'markersUsed': mrks,
        'n_estimates': n_estimates
    }

    return sc


def quickMarkers(
        toc: sparse.csr_matrix,
        clusters: np.ndarray,
        N: Optional[int] = 10,
        FDR: float = 0.01,
        expressCut: float = 0.9,
        verbose: bool = True,
        gene_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Exact R implementation of quickMarkers using sparse matrix indices.
    """
    from scipy import sparse
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    # Convert to coordinate format like R's TsparseMatrix
    toc_coo = toc.tocoo()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(toc.shape[0])]

    # R: w = which(toc@x>expressCut)
    w = toc_coo.data > expressCut

    # R: clCnts = table(clusters)
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    clCnts = dict(zip(unique_clusters, cluster_counts))

    # R: nObs = split(factor(rownames(toc))[toc@i[w]+1],clusters[toc@j[w]+1])
    # R: nObs = sapply(nObs,table)

    # Get gene indices and cluster indices for values > expressCut
    gene_indices = toc_coo.row[w]  # R: toc@i[w]+1 (but Python is 0-indexed)
    cell_indices = toc_coo.col[w]  # R: toc@j[w]+1
    cell_clusters = clusters[cell_indices]  # clusters for those cells

    # Create the split structure - for each cluster, count gene occurrences
    nObs = {}
    for cluster in unique_clusters:
        cluster_mask = cell_clusters == cluster
        if np.any(cluster_mask):
            genes_in_cluster = gene_indices[cluster_mask]
            # Count occurrences of each gene in this cluster
            unique_genes, gene_counts = np.unique(genes_in_cluster, return_counts=True)
            nObs[cluster] = dict(zip(unique_genes, gene_counts))
        else:
            nObs[cluster] = {}

    # Convert to matrix form - genes x clusters
    n_genes = toc.shape[0]
    nObs_matrix = np.zeros((n_genes, len(unique_clusters)))

    for j, cluster in enumerate(unique_clusters):
        for gene_idx, count in nObs[cluster].items():
            nObs_matrix[gene_idx, j] = count

    # R: nTot = rowSums(nObs)
    nTot = nObs_matrix.sum(axis=1)

    # R: tf = t(t(nObs)/as.integer(clCnts[colnames(nObs)]))
    cluster_sizes = np.array([clCnts[cluster] for cluster in unique_clusters])
    tf = nObs_matrix / cluster_sizes[np.newaxis, :]  # Broadcasting

    # R: idf = log(ncol(toc)/nTot)
    n_cells_total = toc.shape[1]
    idf = np.log(n_cells_total / np.maximum(nTot, 1))  # Avoid division by zero

    # R: score = tf*idf
    score = tf * idf[:, np.newaxis]  # Broadcasting idf across clusters

    # Calculate hypergeometric p-values for each gene/cluster combination
    markers = []

    for j, cluster in enumerate(unique_clusters):
        n_cells_cluster = clCnts[cluster]

        for gene_idx in range(n_genes):
            if nObs_matrix[gene_idx, j] == 0:
                continue  # Skip genes not expressed in this cluster

            # Hypergeometric test parameters
            M = n_cells_total  # Total population
            n = int(nTot[gene_idx])  # Success states in population
            N_sample = n_cells_cluster  # Sample size
            k = int(nObs_matrix[gene_idx, j])  # Observed successes

            if n == 0 or k == 0:
                p_val = 1.0
            else:
                # R: phyper(k-1, n, M-n, N_sample, lower.tail=FALSE)
                p_val = stats.hypergeom.sf(k - 1, M, n, N_sample)

            markers.append({
                'gene': gene_names[gene_idx],
                'cluster': str(cluster),
                'tfidf': score[gene_idx, j],
                'geneFrequency': tf[gene_idx, j],
                'geneFrequencyGlobal': nTot[gene_idx] / n_cells_total,
                'p_value': p_val,
                'gene_idx': gene_idx,
                'cluster_idx': j
            })

    if len(markers) == 0:
        return pd.DataFrame()

    # Convert to DataFrame and apply FDR correction per cluster
    df = pd.DataFrame(markers)

    # Apply FDR correction within each cluster
    for cluster in unique_clusters:
        cluster_mask = df['cluster'] == str(cluster)
        if np.any(cluster_mask):
            p_vals = df.loc[cluster_mask, 'p_value'].values
            if len(p_vals) > 0:
                _, p_adjusted, _, _ = multipletests(p_vals, alpha=FDR, method='fdr_bh')
                df.loc[cluster_mask, 'p_adjusted'] = p_adjusted

    # Filter significant markers
    if 'p_adjusted' in df.columns:
        df = df[df['p_adjusted'] < FDR]

    # Sort by tfidf and limit N per cluster
    result_markers = []
    for cluster in unique_clusters:
        cluster_markers = df[df['cluster'] == str(cluster)].copy()
        if len(cluster_markers) > 0:
            cluster_markers = cluster_markers.sort_values('tfidf', ascending=False)
            if N is not None:
                cluster_markers = cluster_markers.head(N)
            result_markers.append(cluster_markers)

    if result_markers:
        final_df = pd.concat(result_markers, ignore_index=True)
        # Remove helper columns
        final_df = final_df.drop(['gene_idx', 'cluster_idx'], axis=1, errors='ignore')
        return final_df
    else:
        return pd.DataFrame()


def estimateNonExpressingCells(
        sc: "SoupChannel",
        nonExpressedGeneList: list,
        clusters: Optional[np.ndarray] = None,
        maximumContamination: float = 1.0,
        FDR: float = 0.05,
        verbose: bool = True
) -> np.ndarray:
    """
    Exact R implementation working with gene sets at cluster level.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if clusters is None:
        if 'clusters' in sc.metaData.columns:
            clusters = sc.metaData['clusters'].values
        else:
            clusters = np.array(['0'] * sc.n_cells)

    unique_clusters = np.unique(clusters)

    # R: tgtGns = unique(unlist(nonExpressedGeneList))
    tgtGns = list(set(nonExpressedGeneList))  # For single gene list, this is just the list

    # Get gene indices
    gene_indices = []
    for gene in tgtGns:
        try:
            if hasattr(sc.soupProfile, 'index'):
                gene_idx = sc.soupProfile.index.get_loc(gene)
                gene_indices.append(gene_idx)
            else:
                gene_indices.append(int(gene))
        except (KeyError, ValueError):
            if verbose:
                print(f"Warning: Gene {gene} not found in data")
            continue

    if len(gene_indices) == 0:
        return np.ones(sc.n_cells, dtype=bool)

    # R: dat = sc$toc[tgtGns,,drop=FALSE]
    dat = sc.toc[gene_indices, :]

    # R: cnts = do.call(rbind,lapply(nonExpressedGeneList,function(e) colSums(dat[e,,drop=FALSE])))
    # For single gene lists, this is just the gene counts
    cnts = dat  # For single gene case, cnts is just the gene counts

    # R: exp = outer(sc$soupProfile[tgtGns,'est'],sc$metaData$nUMIs*maximumContamination)
    soup_est = sc.soupProfile.iloc[gene_indices]['est'].values
    cell_nUMIs = sc.metaData['nUMIs'].values
    exp = np.outer(soup_est, cell_nUMIs * maximumContamination)

    # R: s = split(names(clusters),clusters)
    cluster_cell_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_cell_dict:
            cluster_cell_dict[cluster] = []
        cluster_cell_dict[cluster].append(i)

    # R: clustExp = ppois(cnts-1,exp,lower.tail=FALSE)
    # Convert sparse to dense for easier manipulation
    cnts_dense = cnts.toarray() if hasattr(cnts, 'toarray') else cnts

    # Calculate p-values for each gene/cell combination
    clustExp = np.zeros_like(cnts_dense, dtype=float)
    for i in range(cnts_dense.shape[0]):
        for j in range(cnts_dense.shape[1]):
            if exp[i, j] <= 0:
                clustExp[i, j] = 0.0 if cnts_dense[i, j] > 0 else 1.0
            else:
                # R: ppois(cnts-1, exp, lower.tail=FALSE)
                clustExp[i, j] = 1 - stats.poisson.cdf(cnts_dense[i, j] - 1, exp[i, j])

    # R: clustExp = t(apply(clustExp,1,p.adjust,method='BH'))
    # Apply FDR correction across cells for each gene
    for gene_idx in range(clustExp.shape[0]):
        _, clustExp[gene_idx, :], _, _ = multipletests(
            clustExp[gene_idx, :], alpha=FDR, method='fdr_bh'
        )

    # R: clustExp = do.call(rbind,lapply(s,function(e) apply(clustExp[,e,drop=FALSE],1,min)))
    # For each cluster, get minimum p-value across cells in that cluster for each gene
    cluster_pvals = np.zeros((len(gene_indices), len(unique_clusters)))

    for j, cluster in enumerate(unique_clusters):
        cell_indices_in_cluster = cluster_cell_dict[cluster]
        if len(cell_indices_in_cluster) > 0:
            # Get minimum p-value across cells in this cluster for each gene
            cluster_subset = clustExp[:, cell_indices_in_cluster]
            cluster_pvals[:, j] = np.min(cluster_subset, axis=1)
        else:
            cluster_pvals[:, j] = 1.0

    # R: clustExp = clustExp>=FDR
    cluster_passes = cluster_pvals >= FDR

    # R: clustExp = clustExp[match(clusters,rownames(clustExp)),,drop=FALSE]
    # Expand back to cell level - if cluster passes for a gene, all cells in cluster pass
    use_cells = np.ones(sc.n_cells, dtype=bool)

    for i, cluster in enumerate(clusters):
        cluster_idx = np.where(unique_clusters == cluster)[0]
        if len(cluster_idx) > 0:
            cluster_idx = cluster_idx[0]
            # If ANY gene fails the test for this cluster, exclude all cells in cluster
            if not np.all(cluster_passes[:, cluster_idx]):
                use_cells[i] = False

    return use_cells
