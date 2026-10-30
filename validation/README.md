# Validation against R SoupX

Reproduces the numbers and figure in the main README.

**Data** (not included; CC BY 4.0, 10x Genomics): *10k 1:1 Mixture of Human HEK293T and Mouse NIH3T3 cells,
3' v3.1, Chromium Controller* (Cell Ranger 6.1.0),
https://10xgenomics.com/datasets/10-k-1-1-mixture-of-human-hek-293-t-and-mouse-nih-3-t-3-cells-3-v-3-1-chromium-controller-3-1-standard-6-1-0
Files: `10k_hgmm_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5` and
`..._raw_feature_bc_matrix.h5`.

**Requirements**: soupx-python with `scanpy`, `igraph`, `matplotlib`; R with `Matrix` and `SoupX` (1.6.2 used).

```bash
# run inside validation/
python prepare_hgmm.py FILTERED_H5 RAW_H5 OUT   # species calls, Leiden clusters, export (~2 min)
Rscript run_soupx.R OUT                          # R reference incl. timings (~10 min, mostly multinomial)
python parity.py OUT                             # Python run, parity, ground truth -> results.json, validation.png
```

`prepare_hgmm.py` classifies cells by human UMI fraction (> 0.9 human, < 0.1 mouse) and drops the rest (mostly
doublets). Both implementations get identical inputs: counts, unique gene IDs, cluster labels, and the empty
droplets (0 < UMIs < 100, the only droplets SoupX uses). Cluster labels are written as `c00`, `c01`, ... so
that R and Python sort them identically regardless of locale.

**Ground truth.** Mouse counts in human cells (and vice versa) are ambient. Divided by the share of the other
species in the soup profile they estimate the total contamination ρ. For each correction method the script
reports the fraction of cross-species counts removed and the cross-species share of all removed counts,
compared with the share expected if every removed count were ambient.
