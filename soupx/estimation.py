"""Contamination estimation, ported from SoupX 1.6.2 (R/autoEstCont.R, R/quickMarkers.R,
R/estimateNonExpressingCells.R)."""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse, stats


def _p_adjust_bh(p, axis=-1):
    """Benjamini-Hochberg adjustment, equivalent to R's p.adjust(method='BH')."""
    return stats.false_discovery_control(np.clip(p, 0, 1), method="bh", axis=axis)


def _onehot(clusters):
    labels, idx = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    G = sparse.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, labels.size))
    return labels, idx, G


def quickMarkers(toc, clusters, N=10, FDR=0.01, expressCut=0.9, verbose=True, gene_names=None):
    """tf-idf marker genes per cluster (R quickMarkers). toc is genes x cells; N=None means all.

    As in R, genes are ranked within a cluster by tf-idf with ties broken by gene name, and
    q-values are BH-adjusted over all genes. If at least N genes pass FDR, the top N by tf-idf
    are returned without the FDR filter (R behaviour).
    """
    toc = sparse.csr_matrix(toc)
    genes = np.asarray(gene_names if gene_names is not None else [f"gene_{i}" for i in range(toc.shape[0])], dtype=str)
    labels, idx, G = _onehot(clusters)
    nCells = toc.shape[1]
    nObs = np.asarray(((toc > expressCut).astype(np.float64) @ G).todense())
    clCnts = np.bincount(idx).astype(np.float64)
    nTot = nObs.sum(1)
    tf = nObs / clCnts
    ntf = (nTot[:, None] - nObs) / (nCells - clCnts)
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = np.log(nCells / nTot)
        score = tf * idf[:, None]
    # hypergeometric p-values, evaluated once per distinct (nObs, nTot, clCnt) triple (identical values)
    # (p = 1 where nObs = 0, as R's phyper(-1, ...))
    p = np.ones_like(nObs)
    nz = nObs > 0
    trip = np.stack([nObs[nz], np.broadcast_to(nTot[:, None], nObs.shape)[nz],
                     np.broadcast_to(clCnts, nObs.shape)[nz]], axis=1)
    u, inv = np.unique(trip, axis=0, return_inverse=True)
    p[nz] = stats.hypergeom.sf(u[:, 0] - 1, nCells, u[:, 1], u[:, 2])[inv.ravel()]
    qvals = _p_adjust_bh(p, axis=0)
    rowOrder = np.argsort(genes, kind="stable")  # R: rows of table(factor(rownames)) are sorted names
    N = np.inf if N is None else N
    rows = []
    for e, lab in enumerate(labels):
        o = rowOrder[np.argsort(-score[rowOrder, e], kind="stable")]  # NaN last, like R
        sel = o[:int(N)] if (qvals[:, e] < FDR).sum() >= N else o[qvals[o, e] < FDR]
        other = np.delete(np.arange(labels.size), e)
        snd = other[np.argmax(tf[np.ix_(sel, other)], axis=1)] if other.size else np.full(sel.size, e)
        rows.append(pd.DataFrame({
            "gene": genes[sel], "cluster": lab, "geneFrequency": tf[sel, e],
            "geneFrequencyOutsideCluster": ntf[sel, e], "geneFrequencySecondBest": tf[sel, snd],
            "geneFrequencyGlobal": nTot[sel] / nCells, "secondBestClusterName": labels[snd],
            "tfidf": score[sel, e], "idf": idf[sel], "qval": qvals[sel, e]}))
    return pd.concat(rows, ignore_index=True)


def _non_expressing(sc, sets, clusters, maximumContamination, FDR):
    """cells x sets boolean matrix: can the cell's cluster be used to estimate rho from this set."""
    gidx = pd.Index(sc.soupProfile.index)
    toc = sparse.csr_matrix(sc.toc)
    S = sparse.csr_matrix([np.isin(np.arange(toc.shape[0]), gidx.get_indexer(g)) for g in sets], dtype=np.float64)
    cnts = np.asarray((S @ toc).todense())
    exp = np.outer(S @ sc.soupProfile["est"].to_numpy(), sc.metaData["nUMIs"].to_numpy() * maximumContamination)
    p = _p_adjust_bh(stats.poisson.sf(cnts - 1, exp), axis=1)
    labels, idx, _ = _onehot(clusters)
    passes = np.stack([p[:, idx == k].min(axis=1) >= FDR for k in range(labels.size)], axis=1)
    return passes[:, idx].T


def estimateNonExpressingCells(sc, nonExpressedGeneList, clusters=None, maximumContamination=1.0,
                               FDR=0.05, verbose=True):
    """Cells whose cluster shows no evidence of expressing each gene set (R estimateNonExpressingCells).

    nonExpressedGeneList: dict {name: [genes]} -> DataFrame (cells x sets); a flat list of genes is
    treated as one set -> boolean array (cells). Without clusters every cell is its own cluster.
    """
    if clusters is None:
        clusters = sc.metaData["clusters"].to_numpy() if "clusters" in sc.metaData.columns else False
    if clusters is False:
        if verbose:
            print("No clusters found or supplied, using every cell as its own cluster.")
        clusters = np.arange(sc.toc.shape[1])
    flat = not isinstance(nonExpressedGeneList, dict) and all(isinstance(g, str) for g in nonExpressedGeneList)
    sets = {"set": list(nonExpressedGeneList)} if flat else dict(nonExpressedGeneList)
    out = _non_expressing(sc, list(sets.values()), clusters, maximumContamination, FDR)
    if out.sum() == 0:
        warnings.warn("No non-expressing cells identified. Consider increasing FDR or maximumContamination.")
    return out[:, 0] if flat else pd.DataFrame(out, index=sc.metaData.index, columns=list(sets))


def autoEstCont(sc, topMarkers=None, tfidfMin=1.0, soupQuantile=0.90, maxMarkers=100,
                contaminationRange=(0.01, 0.8), rhoMaxFDR=0.2, priorRho=0.05, priorRhoStdDev=0.10,
                forceAccept=False, verbose=True, **kwargs):
    """Estimate the global contamination fraction from cluster-specific markers (R autoEstCont)."""
    if "clusters" not in sc.metaData.columns:
        raise ValueError("Clustering information must be supplied, run setClusters first.")
    clusters = sc.metaData["clusters"].to_numpy().astype(str)
    labels, idx, G = _onehot(clusters)
    ctoc = np.asarray((sparse.csr_matrix(sc.toc) @ G).todense())   # genes x clusters
    cn = ctoc.sum(0)
    soupProf = sc.soupProfile.sort_values("est", ascending=False, kind="stable")
    soupMin = np.quantile(soupProf["est"].to_numpy(), soupQuantile)
    if topMarkers is None:
        mrks = quickMarkers(sc.toc, clusters, N=None, gene_names=list(sc.soupProfile.index))
        mrks = mrks.assign(_neg=-mrks["tfidf"]).sort_values(["gene", "_neg"], kind="stable")
        mrks = mrks[~mrks["gene"].duplicated()]
        mrks = mrks.sort_values("_neg", kind="stable").drop(columns="_neg")
        mrks = mrks[mrks["tfidf"] > tfidfMin]
    else:
        mrks = topMarkers
    filtPass = mrks[mrks["gene"].isin(soupProf.index[soupProf["est"] > soupMin])]
    tgts = list(filtPass["gene"].head(maxMarkers))
    if verbose:
        print(f"{len(mrks)} genes passed tf-idf cut-off and {len(filtPass)} soup quantile filter. "
              f"Taking the top {len(tgts)}.")
    if not tgts:
        raise ValueError("No plausible marker genes found. Is the channel low complexity? "
                         "If not, reduce tfidfMin or soupQuantile")
    if len(tgts) < 10:
        warnings.warn("Fewer than 10 marker genes found. Consider reducing tfidfMin or soupQuantile")
    ute = _non_expressing(sc, [[g] for g in tgts], clusters, max(contaminationRange), rhoMaxFDR)
    ute = ute[[np.flatnonzero(idx == k)[0] for k in range(labels.size)]].T   # genes x clusters
    gi = sc.soupProfile.index.get_indexer(tgts)
    expCnts = np.outer(sc.soupProfile["est"].to_numpy()[gi], cn)
    obsCnts = ctoc[gi]
    dd = pd.DataFrame({"gene": np.repeat(tgts, labels.size), "cluster": np.tile(labels, len(tgts)),
                       "passNonExp": ute.ravel(), "rhoEst": (obsCnts / expCnts).ravel(),
                       "obsCnt": obsCnts.ravel(), "expCnt": expCnts.ravel()})
    dd["useEst"] = dd["passNonExp"]
    if dd["useEst"].sum() < 10:
        warnings.warn("Fewer than 10 independent estimates, rho estimation is likely to be unstable.")
    if verbose:
        print(f"Using {int(dd['useEst'].sum())} independent estimates of rho.")
    v2 = (priorRhoStdDev / priorRho) ** 2
    k = 1 + v2 ** -2 / 2 * (1 + np.sqrt(1 + 4 * v2))
    theta = priorRho / (k - 1)
    rhoProbes = np.arange(1001) * 0.001                      # R: seq(0, 1, .001)
    use = dd[dd["useEst"]]
    post = (stats.gamma.pdf(rhoProbes[:, None], a=k + use["obsCnt"].to_numpy(),
                            scale=theta / (1 + theta * use["expCnt"].to_numpy())).mean(axis=1)
            if len(use) else np.zeros_like(rhoProbes))
    w = np.flatnonzero((rhoProbes >= contaminationRange[0]) & (rhoProbes <= contaminationRange[1]))
    rhoEst = rhoProbes[w][np.argmax(post[w])]
    half = rhoProbes[w][post[w] >= post[w].max() / 2]
    if verbose:
        print(f"Estimated global rho of {rhoEst:.2f}")
    sc.fit = {"dd": dd, "priorRho": priorRho, "priorRhoStdDev": priorRhoStdDev, "posterior": post,
              "rhoEst": rhoEst, "rhoFWHM": (half.min(), half.max()), "rhoProbes": rhoProbes, "markersUsed": mrks}
    sc.set_contamination_fraction(rhoEst, forceAccept=forceAccept)
    return sc


def calculateContaminationFraction(sc, nonExpressedGeneList, useToEst, verbose=True, forceAccept=False):
    """Global rho from gene sets known not to be expressed in some cells (R calculateContaminationFraction).

    useToEst: output of estimateNonExpressingCells (cells x sets, or cells for a single flat gene list).
    R fits glm(counts ~ 1, poisson, offset = log(expected soup counts)); its MLE is sum(counts) /
    sum(expected). rhoLow/rhoHigh are the 95% profile-likelihood interval (R: confint).
    """
    flat = not isinstance(nonExpressedGeneList, dict) and all(isinstance(g, str) for g in nonExpressedGeneList)
    sets = [list(nonExpressedGeneList)] if flat else list(dict(nonExpressedGeneList).values())
    use = np.asarray(useToEst, dtype=bool).reshape(len(sc.metaData), -1)
    if not use.any():
        raise ValueError("No cells specified as acceptable for estimation. useToEst must not be all False")
    toc = sparse.csr_matrix(sc.toc)
    nUMIs = sc.metaData["nUMIs"].to_numpy(np.float64)
    obs = exp = 0.0
    for i, g in enumerate(sets):
        gi = sc.soupProfile.index.get_indexer(g)
        w = use[:, i]
        obs += toc[gi][:, w].sum()
        exp += (nUMIs[w] * sc.soupProfile["est"].to_numpy()[gi].sum()).sum()
    rho = obs / exp
    half = stats.chi2.ppf(0.95, 1) / 2
    llr = lambda r: (obs * np.log(obs / (r * exp)) if obs > 0 else 0.0) - (obs - r * exp) - half
    from scipy.optimize import brentq
    lo = brentq(llr, 1e-300, rho) if obs > 0 else 0.0
    hi = brentq(llr, max(rho, 1e-300), 10 * rho + 10 / exp)
    sc.set_contamination_fraction(rho, forceAccept=forceAccept)
    sc.metaData["rhoLow"], sc.metaData["rhoHigh"] = lo, hi
    sc.fit = {"rhoEst": rho, "rhoLow": lo, "rhoHigh": hi, "obsCnt": obs, "expCnt": exp}
    if verbose:
        print(f"Estimated global contamination fraction of {100 * rho:.2f}%")
    return sc
