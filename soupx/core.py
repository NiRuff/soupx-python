"""
Core SoupChannel class for SoupX implementation.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union, Dict, Any


class SoupChannel:
    """
    Container for single-cell RNA-seq data and soup contamination analysis.

    Equivalent to R SoupChannel object. Stores:
    - raw droplet counts (tod = table of droplets)
    - filtered cell counts (toc = table of counts)
    - metadata about cells
    - soup profile (background contamination profile)
    - contamination fraction estimates
    """

    def __init__(
            self,
            raw_counts: sparse.csr_matrix,
            filtered_counts: sparse.csr_matrix,
            raw_barcodes: Optional[np.ndarray] = None,
            filtered_barcodes: Optional[np.ndarray] = None,
            gene_names: Optional[np.ndarray] = None,
            metadata: Optional[pd.DataFrame] = None,
            calc_soup_profile: bool = True,
            **kwargs
    ):
        """
        Initialize SoupChannel object.

        Parameters
        ----------
        raw_counts : sparse matrix
            Raw droplet counts matrix (genes x droplets), includes empty droplets
        filtered_counts : sparse matrix
            Filtered cell counts matrix (genes x cells), cellranger filtered
        raw_barcodes : array-like, optional
            Barcode names for raw droplets
        filtered_barcodes : array-like, optional
            Barcode names for filtered cells
        gene_names : array-like, optional
            Gene names/IDs
        metadata : DataFrame, optional
            Cell metadata with barcodes as index
        calc_soup_profile : bool, default True
            Whether to automatically calculate soup profile
        **kwargs : dict
            Additional metadata to store
        """

        # Input validation
        if raw_counts.shape[0] != filtered_counts.shape[0]:
            raise ValueError("Raw and filtered counts must have same number of genes")

        # Core count matrices
        self.raw_counts = raw_counts.tocsr()  # tod in R
        self.filtered_counts = filtered_counts.tocsr()  # toc in R

        # Dimensions
        self.n_genes = raw_counts.shape[0]
        self.n_raw_droplets = raw_counts.shape[1]
        self.n_cells = filtered_counts.shape[1]

        # Identifiers
        self.gene_names = self._validate_names(gene_names, self.n_genes, 'gene')
        self.raw_barcodes = self._validate_names(raw_barcodes, self.n_raw_droplets, 'raw_barcode')
        self.filtered_barcodes = self._validate_names(filtered_barcodes, self.n_cells, 'cell_barcode')

        # Create metadata DataFrame
        self.metadata = self._init_metadata(metadata)

        # Soup analysis results (will be filled later)
        self.soup_profile = None
        self.contamination_fraction = None
        self.clusters = None

        # Store additional parameters
        self.params = kwargs

        # Calculate soup profile if requested
        if calc_soup_profile:
            from .estimation import estimate_soup
            self = estimate_soup(self)

    def _validate_names(self, names: Optional[np.ndarray], expected_length: int, name_type: str) -> np.ndarray:
        """Validate and create name arrays."""
        if names is None:
            return np.array([f"{name_type}_{i}" for i in range(expected_length)])

        names = np.asarray(names)
        if len(names) != expected_length:
            raise ValueError(f"{name_type} names length ({len(names)}) doesn't match expected ({expected_length})")
        return names

    def _init_metadata(self, metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Initialize cell metadata DataFrame."""

        # Calculate basic cell statistics
        cell_stats = pd.DataFrame(
            index=self.filtered_barcodes,
            data={
                'n_umis': np.array(self.filtered_counts.sum(axis=0)).flatten(),
                'n_genes': np.array((self.filtered_counts > 0).sum(axis=0)).flatten()
            }
        )

        # Merge with provided metadata if available
        if metadata is not None:
            # Ensure metadata index matches cell barcodes
            if not all(bc in metadata.index for bc in self.filtered_barcodes):
                raise ValueError("Metadata index must contain all cell barcodes")

            metadata_aligned = metadata.loc[self.filtered_barcodes]
            # Combine, preferring user metadata for overlapping columns
            for col in metadata_aligned.columns:
                cell_stats[col] = metadata_aligned[col]

        return cell_stats

    @property
    def raw_umi_counts(self) -> np.ndarray:
        """UMI counts per raw droplet."""
        return np.array(self.raw_counts.sum(axis=0)).flatten()

    def set_clusters(self, clusters: Union[np.ndarray, pd.Series, Dict[str, Any]]) -> None:
        """
        Set cluster assignments for cells.

        Parameters
        ----------
        clusters : array-like or dict
            Cluster assignments. If dict, keys should be cell barcodes.
        """
        if isinstance(clusters, dict):
            cluster_series = pd.Series(clusters)
            cluster_array = cluster_series.loc[self.filtered_barcodes].values
        else:
            cluster_array = np.asarray(clusters)

        if len(cluster_array) != self.n_cells:
            raise ValueError(f"Cluster array length ({len(cluster_array)}) doesn't match n_cells ({self.n_cells})")

        self.metadata['clusters'] = cluster_array
        self.clusters = cluster_array

    def set_contamination_fraction(self, contamination_fraction: float) -> None:
        """
        Set global contamination fraction for all cells.

        Parameters
        ----------
        contamination_fraction : float
            Fraction of UMIs from background contamination (0-1)
        """
        if not 0 <= contamination_fraction <= 1:
            raise ValueError("Contamination fraction must be between 0 and 1")

        self.contamination_fraction = contamination_fraction
        self.metadata['rho'] = contamination_fraction

    def __repr__(self) -> str:
        return f"SoupChannel with {self.n_genes} genes and {self.n_cells} cells"