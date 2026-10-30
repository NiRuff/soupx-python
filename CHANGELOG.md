# Changelog

## 0.4.0

Validated against R SoupX 1.6.2 (identical results on a 10x human/mouse mixture, see README).

Fixed
- `adjustCounts`: soup counts of one cluster could be redistributed to the cells of another cluster
  (cluster order differed between aggregation and `expandClusters`).
- `adjustCounts(method="subtraction")` clipped at zero instead of redistributing via `alloc`, removing less
  than ρ; `"soupOnly"` and `"multinomial"` now follow the R algorithms (previously different procedures,
  multinomial stopped after 200 steps).
- `quickMarkers`: BH correction over all genes as in R (previously only expressed genes, giving extra
  markers); ties ordered by gene name.
- `estimateNonExpressingCells`: a dict of gene sets was treated as a single set of its keys.

Added
- `calculateContaminationFraction` (manual ρ from gene sets, with 95% CI).
- `adjustCounts(seed=...)`, `autoEstCont(topMarkers=...)`, `s.fit["rhoFWHM"]`.
- Tests with R reference values, CI on Linux and Windows, `validation/` scripts.

Changed
- `quickMarkers` returns R's columns (`qval`, `geneFrequencySecondBest`, ...) instead of
  `p_value`/`p_adjusted`.
- `estimateNonExpressingCells` with a dict returns a cells x sets DataFrame (flat gene list: boolean array).
- Much faster `autoEstCont` and `adjustCounts`.
- Python >= 3.10; `statsmodels` dropped, `scanpy` optional (`soupx-python[scanpy]`).
- License file (GPL-2) and NOTICE added.

## 0.3.0
- PyPI release.
