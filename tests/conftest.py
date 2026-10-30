import numpy as np
import pytest
import scipy.sparse as sp

import soupx


def simulate(rho=0.05, n_genes=1000, n_clusters=10, cells_per_cluster=40,
             n_markers=10, marker_weight=0.03, n_empty=5000, seed=0):
    """Synthetic channel with a known contamination fraction.

    Each cluster expresses its own block of marker genes. Cells receive a
    fraction ``rho`` of their counts from the soup (mean expression profile),
    empty droplets (< 100 UMIs) are drawn from the soup only.
    """
    rng = np.random.default_rng(seed)
    n_mk = n_clusters * n_markers
    base = rng.gamma(2.0, 1.0, n_genes)
    base[:n_mk] = 0
    base *= (1 - n_markers * marker_weight) / base.sum()
    profiles = []
    for k in range(n_clusters):
        p = base.copy()
        p[k * n_markers:(k + 1) * n_markers] = marker_weight
        profiles.append(p)
    profiles = np.array(profiles)
    clusters = np.repeat(np.arange(n_clusters), cells_per_cluster)
    n_cells = clusters.size
    sizes = rng.integers(1000, 2000, n_cells)
    soup = (profiles[clusters] * sizes[:, None]).sum(0)
    soup /= soup.sum()
    cells = np.array([rng.multinomial(round(n * (1 - rho)), profiles[c]) + rng.multinomial(round(n * rho), soup)
                      for n, c in zip(sizes, clusters)])
    empties = np.array([rng.multinomial(n, soup) for n in rng.integers(5, 90, n_empty)])
    toc = sp.csr_matrix(cells.T.astype(float))
    tod = sp.csr_matrix(np.hstack([cells.T, empties.T]).astype(float))
    genes = [f"G{i:04d}" for i in range(n_genes)]
    markers = {str(k): set(genes[k * n_markers:(k + 1) * n_markers]) for k in range(n_clusters)}
    return dict(toc=toc, tod=tod, genes=genes, clusters=clusters, soup=soup, rho=rho, markers=markers)


@pytest.fixture(scope="session")
def sim():
    return simulate()


@pytest.fixture
def channel(sim):
    sc = soupx.SoupChannel(sim["tod"], sim["toc"], gene_names=sim["genes"])
    sc.setClusters(sim["clusters"])
    return sc


@pytest.fixture(scope="session")
def fitted(sim):
    sc = soupx.SoupChannel(sim["tod"], sim["toc"], gene_names=sim["genes"])
    sc.setClusters(sim["clusters"])
    return soupx.autoEstCont(sc, verbose=False)
