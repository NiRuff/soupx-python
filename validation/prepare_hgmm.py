"""Prepare a 10x human/mouse mixture (cellranger filtered + raw .h5) for the R parity check.

Usage: python prepare_hgmm.py FILTERED_H5 RAW_H5 OUTDIR
Cells: human/mouse by human UMI fraction (>0.9 / <0.1); mixed cells (mostly doublets) are dropped.
Clusters: Leiden on the standard scanpy workflow. Droplets: only those with 0 < UMIs < 100, which is
all SoupX uses the raw matrix for (estimateSoup).
"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from parity import write

filtered, raw, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
a = sc.read_10x_h5(filtered)
genes = a.var["gene_ids"].to_numpy(str)                 # unique, unlike gene symbols
human = pd.Series(genes).str.startswith("GRCh38").to_numpy()
frac = np.asarray(a.X[:, human].sum(1)).ravel() / np.asarray(a.X.sum(1)).ravel()
species = np.select([frac > 0.9, frac < 0.1], ["human", "mouse"], "mixed")
print(pd.Series(species).value_counts().to_dict())
a = a[species != "mixed"].copy()
a.obs["species"] = species[species != "mixed"]
toc = a.X.T.copy()
sc.pp.normalize_total(a); sc.pp.log1p(a); sc.pp.highly_variable_genes(a, n_top_genes=2000)
sc.pp.pca(a, random_state=0); sc.pp.neighbors(a, random_state=0)
sc.tl.leiden(a, flavor="igraph", n_iterations=2, random_state=0)
clusters = ["c%02d" % int(c) for c in a.obs["leiden"]]
print(pd.crosstab(a.obs["species"], pd.Series(clusters, index=a.obs_names, name="cluster")))

r = sc.read_10x_h5(raw)
assert (r.var["gene_ids"].to_numpy(str) == genes).all(), "raw and filtered genes differ"
tod = r.X.T.tocsc()
n = np.asarray(tod.sum(0)).ravel()
tod = tod[:, (n > 0) & (n < 100)]
print(f"{toc.shape[1]} cells, {tod.shape[1]} droplets for the soup profile")
write(out, "toc", toc); write(out, "tod", tod)
for name, v in [("genes", genes), ("cells", a.obs_names), ("clusters", clusters), ("species", a.obs["species"])]:
    pd.Series(np.asarray(v, dtype=str)).to_csv(f"{out}/{name}.txt", index=False, header=False)
