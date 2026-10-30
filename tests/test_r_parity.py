"""Parity with R SoupX 1.6.2 on a small synthetic channel (reference outputs computed with R)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import soupx

REF = dict(np.load(Path(__file__).parent / "data" / "r_reference.npz"))


@pytest.fixture(scope="module")
def ref_channel():
    toc = sp.csc_matrix((REF["toc_data"].astype(float), REF["toc_indices"].astype(np.int32), REF["toc_indptr"]),
                        shape=tuple(REF["toc_shape"])).tocsr()
    genes = [f"G{i:04d}" for i in range(toc.shape[0])]
    sc = soupx.SoupChannel(toc, toc, gene_names=genes, calcSoupProfile=False)
    cnt = REF["soup_counts"]  # est recomputed exactly as R does (CSV round trip shifts ties)
    sc.setSoupProfile(pd.DataFrame({"est": cnt / cnt.sum(), "counts": cnt}, index=genes))
    sc.setClusters(REF["clusters"])
    return soupx.autoEstCont(sc, verbose=False)


def test_markers_match_r(ref_channel):
    mk = soupx.quickMarkers(ref_channel.toc, REF["clusters"], N=None, gene_names=list(ref_channel.soupProfile.index))
    assert list(mk["gene"]) == list(REF["mk_gene"]) and list(mk["cluster"]) == list(REF["mk_cluster"])
    np.testing.assert_allclose(mk["tfidf"], REF["mk_tfidf"], rtol=1e-12)
    np.testing.assert_allclose(mk["qval"], REF["mk_qval"], rtol=1e-8, atol=1e-300)


def test_auto_est_cont_matches_r(ref_channel):
    fit = ref_channel.fit
    assert fit["rhoEst"] == pytest.approx(REF["rho"], abs=1e-12)
    assert list(dict.fromkeys(fit["dd"]["gene"])) == list(dict.fromkeys(REF["dd_gene"]))
    assert fit["dd"]["passNonExp"].sum() == REF["dd_pass"].sum()
    np.testing.assert_allclose(fit["posterior"], REF["posterior"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("method", ["subtraction", "soupOnly", "multinomial"])
def test_adjust_counts_matches_r(ref_channel, method):
    out = soupx.adjustCounts(ref_channel, method=method, verbose=0, seed=0)
    np.testing.assert_allclose(np.asarray(out.sum(1)).ravel(), REF[f"{method}_genes"], atol=1e-8)
    np.testing.assert_allclose(np.asarray(out.sum(0)).ravel(), REF[f"{method}_cells"], atol=1e-8)
