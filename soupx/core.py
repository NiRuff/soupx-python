"""
Core SoupChannel class for SoupX implementation.
Updated with clustering support and additional methods.
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
    - clustering information
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

        # Clustering information (NEW)
        self.clusters = None

        # Reduced dimension coordinates (for plotting)
        self.reduced_dims = None

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
        # Create basic metadata
        base_metadata = pd.DataFrame(
            index=self.filtered_barcodes,
            data={
                'n_umis': np.array(self.filtered_counts.sum(axis=0)).flatten(),
                'n_genes': np.array((self.filtered_counts > 0).sum(axis=0)).flatten()
            }
        )

        # Merge with provided metadata if available
        if metadata is not None:
            # Ensure metadata index matches cell barcodes
            if not metadata.index.equals(pd.Index(self.filtered_barcodes)):
                print("Warning: metadata index doesn't match cell barcodes")

            # Merge, keeping base_metadata for overlapping columns
            for col in metadata.columns:
                if col not in base_metadata.columns:
                    base_metadata[col] = metadata[col]

        return base_metadata

    @property
    def raw_umi_counts(self) -> np.ndarray:
        """Get UMI counts for all raw droplets."""
        return np.array(self.raw_counts.sum(axis=0)).flatten()

    def set_contamination_fraction(self, contamination_fraction: float):
        """
        Set global contamination fraction for all cells.

        Parameters
        ----------
        contamination_fraction : float
            Fraction of UMIs that are contamination (0-1)
        """
        if not 0 <= contamination_fraction <= 1:
            raise ValueError("Contamination fraction must be between 0 and 1")

        self.contamination_fraction = contamination_fraction

        # Also store per-cell contamination in metadata (all same value)
        self.metadata['rho'] = contamination_fraction

        print(f"Set contamination fraction to {contamination_fraction:.3f}")

    def set_clusters(self, clusters: Union[np.ndarray, pd.Series, Dict], cluster_column: str = 'clusters'):
        """
        Set clustering information for cells.

        Parameters
        ----------
        clusters : array-like or dict
            Cluster assignments for each cell. Can be:
            - Array/series with same length as number of cells
            - Dict mapping cell barcodes to cluster IDs
        cluster_column : str, default 'clusters'
            Column name to store clusters in metadata
        """
        if isinstance(clusters, dict):
            # Map from barcodes to cluster IDs
            cluster_array = np.array([clusters.get(bc, -1) for bc in self.filtered_barcodes])
        else:
            cluster_array = np.asarray(clusters)

        if len(cluster_array) != self.n_cells:
            raise ValueError(f"Clusters length ({len(cluster_array)}) doesn't match number of cells ({self.n_cells})")

        self.clusters = cluster_array
        self.metadata[cluster_column] = cluster_array

        n_clusters = len(np.unique(cluster_array[cluster_array >= 0]))
        print(f"Set {n_clusters} clusters for {self.n_cells} cells")

    def set_reduced_dims(self, coords: np.ndarray, coord_names: Optional[list] = None):
        """
        Set reduced dimension coordinates (e.g., UMAP, tSNE).

        Parameters
        ----------
        coords : array-like, shape (n_cells, n_dims)
            Reduced dimension coordinates
        coord_names : list, optional
            Names for coordinate dimensions
        """
        coords = np.asarray(coords)
        if coords.shape[0] != self.n_cells:
            raise ValueError(f"Coordinates shape ({coords.shape}) doesn't match number of cells ({self.n_cells})")

        if coord_names is None:
            coord_names = [f"dim_{i+1}" for i in range(coords.shape[1])]

        self.reduced_dims = pd.DataFrame(
            coords,
            index=self.filtered_barcodes,
            columns=coord_names
        )

        # Also add to metadata
        for i, name in enumerate(coord_names):
            self.metadata[name] = coords[:, i]

    def __repr__(self):
        """String representation of SoupChannel."""
        lines = [
            f"SoupChannel object",
            f"  {self.n_genes} genes x {self.n_cells} cells",
            f"  {self.n_raw_droplets} total droplets ({self.n_raw_droplets - self.n_cells} empty)",
        ]

        if self.soup_profile is not None:
            lines.append(f"  Soup profile: estimated")
        else:
            lines.append(f"  Soup profile: not estimated")

        if self.contamination_fraction is not None:
            lines.append(f"  Contamination fraction: {self.contamination_fraction:.3f}")
        else:
            lines.append(f"  Contamination fraction: not set")

        if self.clusters is not None:
            n_clusters = len(np.unique(self.clusters[self.clusters >= 0]))
            lines.append(f"  Clusters: {n_clusters} clusters")

        return "\n".join(lines)