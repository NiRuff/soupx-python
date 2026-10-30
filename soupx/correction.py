"""Count correction, ported from SoupX 1.6.2 (R/adjustCounts.R, R/utils.R)."""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from typing import Literal


def adjustCounts(sc, clusters=None,
                 method: Literal["subtraction", "soupOnly", "multinomial"] = "subtraction",
                 roundToInt: bool = False, verbose: int = 1, tol: float = 1e-3, pCut: float = 0.01,
                 seed=None) -> sparse.csr_matrix:
    """Remove soup counts from the table of counts (genes x cells), as R's adjustCounts.

    With clusters (default if set), counts are corrected per cluster and the removed soup is
    re-distributed to cells with expandClusters. ``seed`` controls the random parts (stochastic
    rounding, tie breaking in 'multinomial'); R's random streams cannot be reproduced.
    """
    if method not in ("subtraction", "soupOnly", "multinomial"):
        raise ValueError(f"Unknown method: {method}")
    if "rho" not in sc.metaData.columns or sc.metaData["rho"].isna().all():
        raise ValueError("Contamination fractions must have already been calculated/set.")
    rng = np.random.default_rng(seed)
    toc = sparse.csc_matrix(sc.toc, dtype=np.float64)
    nUMIs = sc.metaData["nUMIs"].to_numpy(np.float64)
    rho = sc.metaData["rho"].to_numpy(np.float64)
    soup = sc.soupProfile["est"].to_numpy(np.float64)
    if clusters is None:
        if "clusters" in sc.metaData.columns:
            clusters = sc.metaData["clusters"].to_numpy()
        else:
            print("Warning: Clustering data not found. Adjusting counts at cell level.")
            clusters = False
    if clusters is False:
        out = _adjust(toc, nUMIs, rho, soup, method, pCut, rng, verbose)
    else:
        labels, idx = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
        G = sparse.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, labels.size))
        ctoc = sparse.csc_matrix(toc @ G)
        cn = G.T @ nUMIs
        crho = (G.T @ (nUMIs * rho)) / cn
        if verbose > 0:
            print(f"Adjusting counts with method '{method}' in {labels.size} clusters")
        csoup = ctoc - _adjust(ctoc, cn, crho, soup, method, pCut, rng, verbose)
        out = toc - expandClusters(csoup, toc, idx, nUMIs * rho)
    out = sparse.csr_matrix(out)
    out.eliminate_zeros()
    if roundToInt:
        out.data = np.floor(out.data) + (rng.random(out.data.size) < out.data - np.floor(out.data))
        out.eliminate_zeros()
    return out


def _adjust(toc, nUMIs, rho, soup, method, pCut, rng, verbose):
    """Cell (or cluster) level correction without clusters; returns corrected counts (csc)."""
    if method == "subtraction":
        return toc - _subtraction_soup(toc, nUMIs * rho, soup)
    if method == "soupOnly":
        return _soup_only(toc, nUMIs, rho, soup, pCut)
    return _multinomial(toc, nUMIs, rho, soup, rng)


def _subtraction_soup(toc, expSoup, soupFrac):
    """Soup counts per column: alloc(expSoupCnts, column, soupFrac). Zero entries always get 0."""
    soup = toc.copy()
    for j in range(toc.shape[1]):
        a, b = toc.indptr[j], toc.indptr[j + 1]
        if b > a:
            soup.data[a:b] = alloc(expSoup[j], toc.data[a:b], soupFrac[toc.indices[a:b]])
    return soup


def _soup_only(toc, nUMIs, rho, soup, pCut):
    """R 'soupOnly': drop whole entries that are consistent with being soup (Fisher combination)."""
    x, i = toc.data, toc.indices
    j = np.repeat(np.arange(toc.shape[1]), np.diff(toc.indptr))
    p = stats.poisson.sf(x - 1, nUMIs[j] * soup[i] * rho[j])
    o = np.lexsort((-p, j))  # by cell, then p decreasing (stable, as R's order)
    xo, jo = x[o], j[o]
    csum = np.cumsum(xo)
    first = np.searchsorted(jo, jo, side="left")  # first entry of each entry's cell
    rTot = csum - (csum - xo)[first]               # running total within cell
    pSoup = stats.poisson.sf(rTot - xo - 1, nUMIs[jo] * rho[jo])
    with np.errstate(divide="ignore"):
        q = stats.chi2.sf(-2 * np.log(p[o] * pSoup), 4)
    keep = np.zeros(x.size, dtype=bool)
    keep[o[q < pCut]] = True
    out = toc.copy()
    out.data = np.where(keep, x, 0.0)
    out.eliminate_zeros()
    return out


def _multinomial(toc, nUMIs, rho, ps, rng):
    """R 'multinomial': greedy maximum-likelihood allocation of round(rho*nUMIs) soup counts per column."""
    soup = _subtraction_soup(toc, nUMIs * rho, ps)          # same sparsity as toc
    corr = toc.data - soup.data                               # initial guess: rounded subtraction
    corr = np.floor(corr) + (rng.random(corr.size) < corr - np.floor(corr))
    fitInit = toc.data - corr
    with np.errstate(divide="ignore"):
        logps = np.log(ps)
    for j in range(toc.shape[1]):
        a, b = toc.indptr[j], toc.indptr[j + 1]
        if b == a:
            continue
        genes = toc.indices[a:b]
        lims = toc.data[a:b]
        fit = fitInit[a:b].copy()
        soup.data[a:b] = _fit_multinomial(fit, lims, logps[genes], round(rho[j] * nUMIs[j]), rng)
    return toc - soup


def _fit_multinomial(fit, lims, logps, tgtN, rng):
    """One column of R's multinomial loop (only genes with counts; others cannot change)."""
    for _ in range(int(10 * lims.sum()) + 10):
        inc, dec = fit < lims, fit > 0
        delInc = np.where(inc, logps - np.log(fit + 1), -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            delDec = np.where(dec, -logps + np.log(fit), -np.inf)
        mInc, mDec = delInc.max(), delDec.max()
        wIncAll, wDecAll = np.flatnonzero(inc & (delInc == mInc)), np.flatnonzero(dec & (delDec == mDec))
        curN = fit.sum()
        if (curN < tgtN and not wIncAll.size) or (curN > tgtN and not wDecAll.size):
            break
        if curN < tgtN:
            fit[rng.choice(wIncAll)] += 1
        elif curN > tgtN:
            fit[rng.choice(wDecAll)] -= 1
        else:
            delTot = mInc + mDec
            if delTot == 0:
                fit[wDecAll] -= 1
                zb = np.union1d(wIncAll, wDecAll)
                fit[zb] += wDecAll.size / zb.size
                break
            if not delTot > 0 or not (wIncAll.size and wDecAll.size):
                break
            fit[rng.choice(wIncAll)] += 1
            fit[rng.choice(wDecAll)] -= 1
    return fit


def expandClusters(clustSoupCnts, cellObsCnts, idx, cellWeights, verbose=0):
    """Distribute cluster-level soup counts to cells (R expandClusters).

    clustSoupCnts: genes x clusters; cellObsCnts: genes x cells; idx: cluster index per cell.
    """
    cellObsCnts = sparse.csc_matrix(cellObsCnts, dtype=np.float64)
    clustSoupCnts = sparse.csc_matrix(clustSoupCnts)
    blocks = []
    for k in range(clustSoupCnts.shape[1]):
        cells = np.flatnonzero(idx == k)
        ww = cellWeights[cells] / cellWeights[cells].sum()
        lims = sparse.coo_matrix(cellObsCnts[:, cells])
        nSoup = clustSoupCnts[:, k].toarray().ravel()
        rowTot = np.bincount(lims.row, weights=lims.data, minlength=nSoup.size)
        x = lims.data.copy()                      # case 2: all counts are soup
        x[nSoup[lims.row] == 0] = 0               # case 1: no soup for gene
        todo = np.flatnonzero((nSoup[lims.row] > 0) & (nSoup[lims.row] < rowTot[lims.row]))
        todo = todo[np.argsort(lims.row[todo], kind="stable")]
        g = np.unique(lims.row[todo], return_inverse=True)[1]
        # alloc, vectorized for the common case where proportional shares fit under the limits
        w = ww[lims.col[todo]]
        share = nSoup[lims.row[todo]] * (w / np.bincount(g, weights=w)[g])
        capped = np.bincount(g, weights=share > lims.data[todo]) > 0
        x[todo] = share
        hard = todo[capped[g]]
        for e in np.split(hard, np.flatnonzero(np.diff(lims.row[hard])) + 1) if hard.size else []:
            x[e] = alloc(nSoup[lims.row[e[0]]], lims.data[e], ww[lims.col[e]])
        blocks.append((cells[lims.col], lims.row, x))
    c, r, x = (np.concatenate(v) for v in zip(*blocks))
    return sparse.csc_matrix((x, (r, c)), shape=cellObsCnts.shape)


def alloc(tgt, bucketLims, ws=None):
    """Distribute tgt counts over buckets proportional to ws without exceeding bucketLims (R alloc)."""
    bucketLims = np.asarray(bucketLims, dtype=np.float64)
    ws = np.full(bucketLims.size, 1.0 / bucketLims.size) if ws is None else np.asarray(ws, dtype=np.float64)
    ws = ws / ws.sum()
    if np.all(tgt * ws <= bucketLims):
        return tgt * ws
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.argsort(bucketLims / ws, kind="stable")
        w, y = ws[o], bucketLims[o]
        cw = np.r_[0, np.cumsum(w[:-1])]
        cy = np.r_[0, np.cumsum(y[:-1])]
        k = y / w * (1 - cw) + cy
        k[w == 0] = np.inf
        b = k <= tgt
        resid = tgt - y[b].sum()
        w = w / (1 - w[b].sum())
        out = np.where(b, y, resid * w)
    res = np.empty_like(out)
    res[o] = out
    return res
