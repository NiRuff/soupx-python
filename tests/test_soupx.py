import numpy as np
import pytest

import soupx


def test_soup_profile(channel, sim):
    est = channel.soupProfile["est"].values
    assert est.sum() == pytest.approx(1.0)
    assert np.corrcoef(est, sim["soup"])[0, 1] > 0.99


def test_quick_markers_find_planted_markers(sim):
    mk = soupx.quickMarkers(sim["toc"], sim["clusters"].astype(str), N=5,
                            verbose=False, gene_names=sim["genes"])
    for cl, d in mk.groupby("cluster"):
        assert set(d["gene"]) <= sim["markers"][cl]


def test_auto_est_cont_recovers_rho(fitted, sim):
    assert fitted.fit["rhoEst"] == pytest.approx(sim["rho"], abs=0.01)
    assert (fitted.metaData["rho"] == fitted.fit["rhoEst"]).all()


def test_auto_est_cont_needs_clusters(sim):
    sc = soupx.SoupChannel(sim["tod"], sim["toc"], gene_names=sim["genes"])
    with pytest.raises(ValueError, match="setClusters"):
        soupx.autoEstCont(sc, verbose=False)


@pytest.mark.parametrize("method", ["subtraction", "soupOnly", "multinomial"])
def test_adjust_counts(fitted, sim, method):
    X = sim["toc"].toarray()
    out = soupx.adjustCounts(fitted, method=method, verbose=0).toarray()
    assert out.shape == X.shape
    assert out.min() >= -1e-9 and (out - X).max() <= 1e-9
    removed = 1 - out.sum() / X.sum()
    if method == "soupOnly":  # only removes whole genes it is confident about
        assert 0 < removed < fitted.fit["rhoEst"]
    else:
        assert removed == pytest.approx(fitted.fit["rhoEst"], abs=0.01)


def test_round_to_int(fitted):
    out = soupx.adjustCounts(fitted, method="subtraction", roundToInt=True, verbose=0)
    assert np.allclose(out.data, np.round(out.data))
