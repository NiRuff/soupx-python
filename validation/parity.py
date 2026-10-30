"""soupx-python vs R SoupX on identical inputs, plus cross-species ground truth.

Usage: python parity.py OUTDIR [--figure-only]   (after prepare_hgmm.py and run_soupx.R;\nwrites results.json and validation.png; --figure-only redraws the figure from results.json)
"""
import json, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import soupx

METHODS = ["subtraction", "soupOnly", "multinomial"]


def write(d, name, M):
    M = sp.csc_matrix(M); M.sort_indices()
    for s, v in [("i", M.indices), ("p", M.indptr), ("x", M.data)]:
        v.astype(np.int32).tofile(f"{d}/{name}_{s}.bin")
    np.savetxt(f"{d}/{name}_dim.txt", [*M.shape, M.nnz], fmt="%d")


def read(d, name, xtype=np.float64):
    nr, nc, nnz = np.loadtxt(f"{d}/{name}_dim.txt", dtype=np.int64)
    rd = lambda s, t, n: np.fromfile(f"{d}/{name}_{s}.bin", dtype=t, count=n)
    return sp.csc_matrix((rd("x", xtype, nnz), rd("i", np.int32, nnz), rd("p", np.int32, nc + 1)), shape=(nr, nc))


def lines(d, name):
    return pd.read_csv(f"{d}/{name}.txt", header=None, dtype=str)[0].to_numpy()


def ground_truth(toc, Y, human_gene, human_cell, soup):
    """Cross-species counts are known ambient; scaled by the soup share of the other species."""
    def cross(M):
        M = sp.csc_matrix(M)
        h, m = (np.asarray(M[g].sum(0)).ravel() for g in (human_gene, ~human_gene))
        return np.where(human_cell, m, h)
    other = np.where(human_cell, soup[~human_gene].sum(), soup[human_gene].sum())
    n, c0 = np.asarray(toc.sum(0)).ravel(), cross(toc)
    out = {"rho_true": c0.sum() / (n * other).sum()}
    for m, M in Y.items():
        rem, c1 = n - np.asarray(M.sum(0)).ravel(), cross(M)
        out[m] = {"cross_removed": 1 - c1.sum() / c0.sum(), "cross_share_of_removed": (c0 - c1).sum() / rem.sum(),
                  "expected_share": (rem * other).sum() / rem.sum()}
    return out


def run(d):
    toc, tod = read(d, "toc", np.int32).astype(np.float64), read(d, "tod", np.int32).astype(np.float64)
    genes, clusters, species = lines(d, "genes"), lines(d, "clusters"), lines(d, "species")
    sc = soupx.SoupChannel(sp.csr_matrix(tod), sp.csr_matrix(toc), gene_names=list(genes))
    sc.setClusters(clusters)
    r_soup = pd.read_csv(f"{d}/r_soup.csv", index_col=0)["est"].to_numpy()
    r_mk, r_dd = pd.read_csv(f"{d}/r_markers.csv"), pd.read_csv(f"{d}/r_dd.csv")
    t = time.perf_counter(); soupx.autoEstCont(sc, verbose=False); t_py = {"autoEstCont": time.perf_counter() - t}
    key = lambda m: set(zip(m["gene"].astype(str), m["cluster"].astype(str)))
    a, b = key(r_mk), key(soupx.quickMarkers(sc.toc, clusters, N=None, gene_names=list(genes)))
    res = {"soup_max_abs_diff": float(np.abs(sc.soupProfile["est"].to_numpy() - r_soup).max()),
           "markers_R": len(a), "markers_Py": len(b), "markers_jaccard": len(a & b) / len(a | b),
           "marker_genes_shared": len(set(r_dd["gene"]) & set(sc.fit["dd"]["gene"])), "marker_genes_R": r_dd["gene"].nunique(),
           "rho_R": float(open(f"{d}/r_rho.txt").read()), "rho_Py": float(sc.fit["rhoEst"])}
    r_post = pd.read_csv(f"{d}/r_post.csv")["post"].to_numpy()
    res["posterior_max_rel_diff"] = float(np.abs(r_post - sc.fit["posterior"]).max() / r_post.max())
    Y = {}
    for m in METHODS:
        t = time.perf_counter(); Y[m] = sp.csc_matrix(soupx.adjustCounts(sc, method=m, verbose=0, seed=0)); t_py[m] = time.perf_counter() - t
        r = read(d, f"r_{m}"); diff = abs(Y[m] - r); rem_r = toc.sum() - r.sum()
        res[m] = {"removed_R": rem_r / toc.sum(), "removed_Py": 1 - Y[m].sum() / toc.sum(),
                  "max_abs_diff": float(diff.max()), "sum_abs_diff_rel": float(diff.sum() / rem_r)}
    t_r = pd.read_csv(f"{d}/r_timings.csv").set_index("step")["seconds"].to_dict()
    res["seconds"] = {s: {"R": t_r[s], "Py": t_py[s]} for s in t_py}
    res["ground_truth"] = ground_truth(toc, Y, pd.Series(genes).str.startswith("GRCh38").to_numpy(), species == "human",
                                       sc.soupProfile["est"].to_numpy())
    res["versions"] = {"soupx-python": soupx.__version__, "R": open(f"{d}/r_session.txt").read().strip().splitlines()}
    return res


def figure(res, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    steps, gt = list(res["seconds"]), res["ground_truth"]
    x, w = np.arange(len(steps)), 0.38
    r, p = (np.array([res["seconds"][s][k] for s in steps]) for k in ("R", "Py"))
    axs[0].bar(x - w / 2, r, w, color="#b0b0b0", label="R SoupX 1.6.2")
    axs[0].bar(x + w / 2, p, w, color="#3b6ea8", label="soupx-python")
    for xi, ri, pi in zip(x, r, p):
        axs[0].text(xi, max(ri, pi) * 1.25, f"{ri / pi:.1f}x" if ri >= pi else f"{pi / ri:.1f}x slower",
                    ha="center", fontsize=7)
    axs[0].set(yscale="log", ylabel="runtime [s]", ylim=(3, max(r.max(), p.max()) * 4))
    axs[0].set_xticks(x, steps)
    axs[0].legend(frameon=False, loc="upper left")
    axs[0].set_title("a  Runtime, identical results", loc="left", fontweight="bold")
    cols = ["#3b6ea8", "#e0a040", "#6aa56a"]
    x = np.arange(len(METHODS))
    for ax, key, title in [(axs[1], "cross_removed", "b  Known ambient removed"),
                           (axs[2], "cross_share_of_removed", "c  Cross-species share of removal")]:
        v = [gt[m][key] for m in METHODS]
        ax.bar(x, v, 0.6, color=cols)
        for xi, vi in zip(x, v):
            ax.text(xi, vi - 0.03, f"{vi:.0%}", ha="center", va="top", fontsize=7, color="white", fontweight="bold")
        ax.set_xticks(x, METHODS)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", fontweight="bold")
    for xi, m in zip(x, METHODS):
        axs[2].hlines(gt[m]["expected_share"], xi - 0.4, xi + 0.4, colors="k", linestyles="--", lw=1,
                      label="expected if all removed\ncounts were ambient" if m == METHODS[0] else None)
    axs[2].legend(frameon=False, loc="upper left", fontsize=7)
    axs[1].text(0.98, 0.98, f"rho SoupX {res['rho_Py']:.3f}\nrho cross-species {gt['rho_true']:.3f}",
                transform=axs[1].transAxes, fontsize=7, ha="right", va="top", color="0.3")
    fig.tight_layout(); fig.savefig(path, dpi=200)


if __name__ == "__main__":
    d = sys.argv[1]
    if "--figure-only" in sys.argv:
        figure(json.load(open(f"{d}/results.json")), f"{d}/validation.png"); sys.exit()
    res = run(d)
    json.dump(res, open(f"{d}/results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    figure(res, f"{d}/validation.png")
