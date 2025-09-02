"""
Core SoupChannel class for SoupX implementation.
Updated to match R SoupX naming conventions and structure.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union, Dict, Any


class SoupChannel:
    """
    Container for single-cell RNA-seq data and soup contamination analysis.

    Matches R SoupChannel object structure with:
    - tod: table of droplets (raw counts)
    - toc: table of counts (filtered cells)
    - metaData: cell metadata (not metadata)
    - soupProfile: background contamination profile
    - clusters: clustering information
    """

    def __init__(
            self,
            tod: sparse.csr_matrix,
            toc: sparse.csr_matrix,
            metaData: Optional[pd.DataFrame] = None,
            calcSoupProfile: bool = True,
            **kwargs
    ):
        """
        Initialize SoupChannel object matching R structure.

        Parameters
        ----------
        tod : sparse.csr_matrix
            Table of droplets - raw counts (genes x droplets)
        toc : sparse.csr_matrix
            Table of counts - filtered cells (genes x cells)
        metaData : pd.DataFrame, optional
            Meta data for cells with rownames matching column names of toc
        calcSoupProfile : bool, default True
            Whether to calculate soup profile automatically
        **kwargs
            Additional named parameters to store
        """
        # Store with R naming conventions
        self.tod = tod
        self.toc = toc
        self.raw_counts = tod  # Keep for backwards compatibility
        self.filtered_counts = toc  # Keep for backwards compatibility

        # Gene information
        self.n_genes = toc.shape[0]
        self.n_cells = toc.shape[1]

        # Store gene names if provided
        self.gene_names = kwargs.get('gene_names', None)
        if self.gene_names is None:
            self.gene_names = [f"gene_{i}" for i in range(self.n_genes)]

        # Initialize metaData with R naming
        if metaData is None:
            # Create default metadata with nUMIs column (not n_umis)
            self.metaData = pd.DataFrame({
                'nUMIs': np.array(toc.sum(axis=0)).flatten()
            }, index=[f"cell_{i}" for i in range(self.n_cells)])
        else:
            self.metaData = metaData
            # Ensure nUMIs column exists
            if 'nUMIs' not in self.metaData.columns:
                self.metaData['nUMIs'] = np.array(toc.sum(axis=0)).flatten()

        # Backwards compatibility
        self.metadata = self.metaData

        # Initialize other attributes
        self.soupProfile = None
        self.soup_profile = None  # Backwards compatibility
        self.clusters = None
        self.DR = None  # Dimension reduction
        self.fit = {}  # Store fitting results

        # Store contamination fraction in metaData as 'rho'
        if 'rho' not in self.metaData.columns:
            self.metaData['rho'] = None

        # Store any additional parameters
        for key, value in kwargs.items():
            if key != 'gene_names':  # Already handled
                setattr(self, key, value)

        # Calculate soup profile if requested
        if calcSoupProfile:
            self._calculate_soup_profile()

    def _calculate_soup_profile(self):
        """Calculate the soup profile from empty droplets - FIXED to match R."""
        # Get UMI counts per droplet
        droplet_umis = np.array(self.tod.sum(axis=0)).flatten()

        # CRITICAL FIX: Match R's default soupRange = c(0, 100)
        # This means > 0 AND < 100, not just < 100
        empty_droplets = (droplet_umis > 0) & (droplet_umis < 100)

        if np.sum(empty_droplets) > 0:
            soup_counts = np.array(self.tod[:, empty_droplets].sum(axis=1)).flatten()
            total_soup = np.sum(soup_counts)

            # FIX: Check if gene_names is not None, not its truthiness
            if self.gene_names is not None:
                gene_index = self.gene_names
            else:
                gene_index = [f"Gene_{i:05d}" for i in range(self.n_genes)]

            self.soupProfile = pd.DataFrame({
                'est': soup_counts / total_soup if total_soup > 0 else np.zeros(self.n_genes),
                'counts': soup_counts
            }, index=gene_index)
        else:
            # No droplets in range
            if self.gene_names is not None:
                gene_index = self.gene_names
            else:
                gene_index = [f"Gene_{i:05d}" for i in range(self.n_genes)]

            self.soupProfile = pd.DataFrame({
                'est': np.zeros(self.n_genes),
                'counts': np.zeros(self.n_genes)
            }, index=gene_index)

        self.soup_profile = self.soupProfile  # Backwards compatibility

    @property
    def contamination_fraction(self):
        """Get contamination fraction (rho) - for backwards compatibility."""
        if 'rho' in self.metaData.columns:
            # Return global rho if all values are the same
            unique_rhos = self.metaData['rho'].dropna().unique()
            if len(unique_rhos) == 1:
                return unique_rhos[0]
            elif len(unique_rhos) > 1:
                # Return mean if cell-specific
                return self.metaData['rho'].mean()
        return None

    @contamination_fraction.setter
    def contamination_fraction(self, value):
        """Set contamination fraction (rho) - for backwards compatibility."""
        self.metaData['rho'] = value

    def set_contamination_fraction(self, contFrac, forceAccept=False):
        """
        Set contamination fraction matching R's setContaminationFraction.

        Parameters
        ----------
        contFrac : float or dict
            Contamination fraction (0-1). Can be constant or cell-specific.
        forceAccept : bool
            Allow very high contamination fractions with warning
        """
        # Validation matching R behavior
        if isinstance(contFrac, (int, float)):
            if contFrac > 1:
                raise ValueError("Contamination fraction greater than 1 detected. This is impossible.")
            if contFrac > 0.5:
                if forceAccept:
                    print(f"Extremely high contamination estimated ({contFrac:.2g}). Proceeding with forceAccept=TRUE.")
                else:
                    raise ValueError(f"Extremely high contamination estimated ({contFrac:.2g}). "
                                   "Set forceAccept=TRUE to proceed.")
            elif contFrac > 0.3:
                print(f"Warning: Estimated contamination is very high ({contFrac:.2g}).")

            self.metaData['rho'] = contFrac
        else:
            # Cell-specific contamination
            for cell_id, rho in contFrac.items():
                if cell_id in self.metaData.index:
                    self.metaData.loc[cell_id, 'rho'] = rho

    def setClusters(self, clusters):
        """
        Set clustering information matching R's setClusters.

        Parameters
        ----------
        clusters : array-like or dict
            Cluster assignments for cells
        """
        if hasattr(clusters, '__len__'):
            if len(clusters) != self.n_cells:
                raise ValueError("Invalid cluster specification. Length must match number of cells.")

            # Convert to string to match R behavior
            self.metaData['clusters'] = [str(c) for c in clusters]
            self.clusters = np.array([str(c) for c in clusters])
        else:
            raise ValueError("Invalid cluster specification.")

        # Check for NAs
        if pd.isna(self.metaData['clusters']).any():
            raise ValueError("NAs found in cluster names.")

    def setSoupProfile(self, soupProfile):
        """
        Manually set soup profile matching R's setSoupProfile.

        Parameters
        ----------
        soupProfile : pd.DataFrame
            DataFrame with 'est' and 'counts' columns
        """
        if 'est' not in soupProfile.columns:
            raise ValueError("est column missing from soupProfile")
        if 'counts' not in soupProfile.columns:
            raise ValueError("counts column missing from soupProfile")

        self.soupProfile = soupProfile
        self.soup_profile = soupProfile  # Backwards compatibility

    def setDR(self, DR, reductName=None):
        """
        Set dimension reduction matching R's setDR.

        Parameters
        ----------
        DR : pd.DataFrame or array-like
            Dimension reduction coordinates (e.g., tSNE, UMAP)
        reductName : str, optional
            Name for the reduction
        """
        DR = pd.DataFrame(DR)
        if DR.shape[1] < 2:
            raise ValueError("Need at least two reduced dimensions.")

        if DR.shape[0] != self.n_cells:
            raise ValueError("DR rows must match number of cells.")

        # Add to metadata
        if reductName:
            col_names = [f"{reductName}_1", f"{reductName}_2"]
        else:
            col_names = ["DR_1", "DR_2"]

        self.metaData[col_names[0]] = DR.iloc[:, 0].values
        self.metaData[col_names[1]] = DR.iloc[:, 1].values
        self.DR = col_names