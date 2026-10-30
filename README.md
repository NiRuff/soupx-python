# soupx-python

[![PyPI version](https://badge.fury.io/py/soupx-python.svg)](https://pypi.org/project/soupx-python/)
[![tests](https://github.com/NiRuff/soupx-python/actions/workflows/tests.yml/badge.svg)](https://github.com/NiRuff/soupx-python/actions/workflows/tests.yml)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://github.com/NiRuff/soupx-python/blob/master/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23804016.svg)](https://doi.org/10.5281/zenodo.23804016)

A Python port of [SoupX](https://github.com/constantAmateur/SoupX) (Young & Behjati, 2020) for removing
ambient RNA contamination from droplet-based single-cell RNA-seq data. It follows SoupX 1.6.2 function by
function and reproduces its results numerically (see [Validation](#validation)).

SoupX estimates the ambient ("soup") expression profile from empty droplets, estimates the contamination
fraction ρ from genes that are clearly not expressed in some clusters, and removes the expected soup counts
from each cell. It needs the raw (unfiltered) droplet matrix.

## Installation

```bash
pip install soupx-python            # numpy, scipy, pandas only
pip install "soupx-python[scanpy]"  # adds scanpy for load10X
```

## Usage with scanpy

```python
import scanpy as sc
import soupx

raw = sc.read_10x_h5("raw_feature_bc_matrix.h5")         # all droplets
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")  # cells; same genes in the same order
# ... cluster adata as usual (on a normalized copy), giving adata.obs["leiden"]

s = soupx.SoupChannel(raw.X.T.tocsr(), adata.X.T.tocsr(),           # genes x droplets / genes x cells, raw counts
                      gene_names=list(adata.var["gene_ids"]))       # unique names
s.setClusters(adata.obs["leiden"].to_numpy())
soupx.autoEstCont(s)                                                # sets rho, details in s.fit
adata.layers["soupx"] = soupx.adjustCounts(s).T.tocsr()             # corrected counts, cells x genes
```

The R function names are kept (`autoEstCont`, `adjustCounts`, `quickMarkers`, ...); snake_case aliases exist
for the main ones. `load10X(path)` reads a Cell Ranger `outs` folder.

**Manual estimation** with genes that some cells should not express (e.g. haemoglobin):

```python
genes = {"HB": ["HBA1", "HBA2", "HBB"]}                 # names as in gene_names
use = soupx.estimateNonExpressingCells(s, genes)          # cells x gene sets
soupx.calculateContaminationFraction(s, genes, use)       # rho with 95% CI in s.metaData
```
or set ρ directly with `s.set_contamination_fraction(0.1)`.

**Correction methods** (`adjustCounts(method=...)`), characterised on a human/mouse mixture below:
- `"subtraction"` (default): removes ρ·nUMIs counts per cell, distributed according to the soup profile.
- `"multinomial"`: maximum-likelihood allocation of the same number of counts; very similar results, slower.
- `"soupOnly"`: only removes whole genes that are consistent with being pure soup; removes less, but almost
  only ambient counts.

## Validation

Compared against R SoupX 1.6.2 (R 4.6.1) on the 10x Genomics dataset *10k 1:1 Mixture of Human HEK293T and
Mouse NIH3T3 cells, 3' v3.1, Chromium Controller* (9,588 single-species cells after removing human/mouse
doublets, 68,886 genes, 14 Leiden clusters), with identical inputs and default parameters. Scripts and
instructions: [`validation/`](https://github.com/NiRuff/soupx-python/tree/master/validation).

| | R SoupX | soupx-python |
|---|---|---|
| soup profile | | identical |
| marker table (`quickMarkers`, 124,792 gene/cluster pairs) | | identical, incl. order |
| ρ (`autoEstCont`, 100 markers, 739 estimates) | 0.017 | 0.017, identical posterior |
| corrected counts, max. abs. difference | | subtraction 1e-12, soupOnly 0, multinomial 7e-13 |

Runtime on the same Windows machine (seconds; R / Python): `autoEstCont` 14 / 20, `subtraction` 62 / 13,
`soupOnly` 19 / 6, `multinomial` 576 / 78.

Cross-species counts (mouse genes in human cells and vice versa) are known ambient RNA. Scaled by the share
of the other species in the soup profile they give ρ ≈ 0.022, so `autoEstCont` underestimates slightly on
this dataset (0.017). `subtraction` and `multinomial` remove 71% of the cross-species counts, `soupOnly` 52%;
of the counts removed by `soupOnly`, 99% are cross-species, versus 45% for the other two (49% would be
expected if everything removed were ambient). These are properties of SoupX itself, as R gives the same
numbers.

![validation](https://raw.githubusercontent.com/NiRuff/soupx-python/master/docs/validation_hgmm10k.png)

`tests/` checks R parity on a small synthetic channel with reference values computed in R, so this is
re-tested on every change.

**Known differences to R**
- Random parts (stochastic rounding with `roundToInt=True`, tie breaking in `"multinomial"`) cannot follow
  R's random number stream; use `seed=` for reproducibility. On the data above this did not change results.
- `calculateContaminationFraction`: ρ is identical; the confidence interval is the exact profile-likelihood
  interval, R interpolates it (relative difference ~1e-6).
- Plotting functions and `autoEstCont(doPlot=...)` are not ported; `s.fit` contains the posterior.

## Citation

Please cite the original method:

> Young, M.D., Behjati, S. SoupX removes ambient RNA contamination from droplet-based single-cell RNA
> sequencing data. *GigaScience* 9, giaa151 (2020). https://doi.org/10.1093/gigascience/giaa151

and, if you use this port, the software itself (doi:10.5281/zenodo.23804016, see [CITATION.cff](https://github.com/NiRuff/soupx-python/blob/master/CITATION.cff)).

Related: [decontx-python](https://github.com/NiRuff/decontx-python), a validated port of DecontX, which does
not need empty droplets.

## License

GPL-2.0, as SoupX (see [LICENSE](https://github.com/NiRuff/soupx-python/blob/master/LICENSE) and [NOTICE](https://github.com/NiRuff/soupx-python/blob/master/NOTICE)).
