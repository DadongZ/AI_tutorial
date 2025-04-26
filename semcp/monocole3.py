# Monocle3 Integration for MCP
# Wrapper functions to connect Python to Monocle3 (R)

import numpy as np
import pandas as pd
import scanpy as sc
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Activate automatic conversion
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Import R libraries
    base = importr('base')
    monocle3 = importr('monocle3')
    singlecellexperiment = importr('SingleCellExperiment')
    
    HAS_RPY2 = True
    logger.info("rpy2 and Monocle3 successfully loaded")
    
except ImportError:
    HAS_RPY2 = False
    logger.warning("rpy2 not installed or Monocle3 not available. Will use scanpy's DPT as fallback")


class Monocle3Wrapper:
    """
    Wrapper for Monocle3 functionality to integrate with Python
    """
    
    @staticmethod
    def check_availability() -> bool:
        """Check if Monocle3 is available through rpy2"""
        return HAS_RPY2
    
    @staticmethod
    def anndata_to_cell_data_set(adata) -> ro.ListVector:
        """
        Convert AnnData object to Monocle3 Cell Data Set object
        
        Args:
            adata: AnnData object
            
        Returns:
            R Cell Data Set object
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Extract data from AnnData
            expression_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            gene_metadata = adata.var.copy()
            cell_metadata = adata.obs.copy()
            
            # Ensure index names are set
            if gene_metadata.index.name is None:
                gene_metadata.index.name = "gene_short_name"
            if cell_metadata.index.name is None:
                cell_metadata.index.name = "cell_id"
                
            # Convert to R objects
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_expression_matrix = numpy2ri.py2rpy(expression_matrix.T)  # Transpose to genes × cells
                r_gene_metadata = pandas2ri.py2rpy(gene_metadata)
                r_cell_metadata = pandas2ri.py2rpy(cell_metadata)
            
            # Create Cell Data Set
            r_row_names = ro.StrVector(gene_metadata.index)
            r_col_names = ro.StrVector(cell_metadata.index)
            
            # Set dimnames for expression matrix
            ro.r('dimnames')(r_expression_matrix)[0] = r_row_names
            ro.r('dimnames')(r_expression_matrix)[1] = r_col_names
            
            # Create CDS object
            cds = monocle3.new_cell_data_set(
                expression_data=r_expression_matrix,
                cell_metadata=r_cell_metadata,
                gene_metadata=r_gene_metadata
            )
            
            return cds
            
        except Exception as e:
            logger.error(f"Error converting AnnData to Cell Data Set: {str(e)}")
            raise
    
    @staticmethod
    def preprocess_cds(cds) -> ro.ListVector:
        """
        Preprocess Cell Data Set (normalize, PCA)
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            Preprocessed Cell Data Set object
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Normalize and preprocess
            cds = monocle3.preprocess_cds(cds, num_dim=50)
            
            return cds
            
        except Exception as e:
            logger.error(f"Error preprocessing Cell Data Set: {str(e)}")
            raise
    
    @staticmethod
    def reduce_dimension(cds) -> ro.ListVector:
        """
        Run dimensionality reduction (UMAP) on Cell Data Set
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            Cell Data Set with reduced dimensions
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Reduce dimension
            cds = monocle3.reduce_dimension(cds)
            
            return cds
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            raise
    
    @staticmethod
    def cluster_cells(cds) -> ro.ListVector:
        """
        Cluster cells in Cell Data Set
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            Cell Data Set with clusters
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Cluster cells
            cds = monocle3.cluster_cells(cds)
            
            return cds
            
        except Exception as e:
            logger.error(f"Error clustering cells: {str(e)}")
            raise
    
    @staticmethod
    def learn_graph(cds) -> ro.ListVector:
        """
        Learn trajectory graph on Cell Data Set
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            Cell Data Set with trajectory graph
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Learn graph
            cds = monocle3.learn_graph(cds)
            
            return cds
            
        except Exception as e:
            logger.error(f"Error learning graph: {str(e)}")
            raise
    
    @staticmethod
    def order_cells(cds, root_cells=None) -> ro.ListVector:
        """
        Order cells along trajectory in Cell Data Set
        
        Args:
            cds: Cell Data Set object
            root_cells: Optional list of root cell IDs
            
        Returns:
            Cell Data Set with cells ordered along trajectory
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Convert root cells to R if provided
            r_root_cells = ro.StrVector(root_cells) if root_cells is not None else ro.r('NULL')
            
            # Order cells
            if root_cells is not None:
                cds = monocle3.order_cells(cds, root_cells=r_root_cells)
            else:
                cds = monocle3.order_cells(cds)
            
            return cds
            
        except Exception as e:
            logger.error(f"Error ordering cells: {str(e)}")
            raise
    
    @staticmethod
    def get_pseudotime(cds) -> np.ndarray:
        """
        Get pseudotime values from Cell Data Set
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            Numpy array of pseudotime values
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Extract pseudotime
            pseudotime = monocle3.pseudotime(cds)
            
            # Convert to numpy array
            with localconverter(ro.default_converter + numpy2ri.converter):
                pseudotime_np = numpy2ri.rpy2py(pseudotime)
                
            return pseudotime_np
            
        except Exception as e:
            logger.error(f"Error getting pseudotime: {str(e)}")
            raise

    @staticmethod
    def find_trajectory_genes(cds) -> pd.DataFrame:
        """
        Find genes that change as a function of pseudotime
        
        Args:
            cds: Cell Data Set object
            
        Returns:
            DataFrame of trajectory-related genes and statistics
        """
        if not HAS_RPY2:
            raise ImportError("rpy2 not installed or Monocle3 not available")
            
        try:
            # Find trajectory-dependent genes
            gene_fits = monocle3.fit_models(cds, model_formula_str=ro.StrVector(["~monocle3_pseudotime"]))
            
            # Get coefficient table
            coef_table = monocle3.coefficient_table(gene_fits)
            
            # Convert to pandas DataFrame
            with localconverter(ro.default_converter + pandas2ri.converter):
                coef_df = pandas2ri.rpy2py(coef_table)
                
            # Filter for significant genes
            sig_genes = coef_df[coef_df['q_value'] < 0.05].copy()
            
            return sig_genes
            
        except Exception as e:
            logger.error(f"Error finding trajectory genes: {str(e)}")
            raise

    @staticmethod
    def run_full_trajectory_analysis(adata, start_cluster=None) -> Dict:
        """
        Run complete Monocle3 trajectory analysis on AnnData object
        
        Args:
            adata: AnnData object
            start_cluster: Optional starting cluster for root cells
            
        Returns:
            Dictionary with trajectory analysis results
        """
        if not HAS_RPY2:
            logger.warning("Monocle3 not available, using scanpy's diffusion pseudotime as fallback")
            # Fallback to scanpy's diffusion pseudotime
            sc.tl.diffmap(adata)
            
            if start_cluster is None:
                # Use cluster with lowest mean diffusion component
                cluster_means = adata.obs.groupby('leiden').mean()['diffmap_1']
                start_cluster = cluster_means.idxmin()
                
            # Find cells in the start cluster
            root_cells = adata.obs[adata.obs['leiden'] == start_cluster].index
            if len(root_cells) == 0:
                raise ValueError(f"No cells found in cluster {start_cluster}")
                
            # Calculate pseudotime
            sc.tl.dpt(adata, root=np.random.choice(root_cells))
            
            # Return results
            return {
                "pseudotime": adata.obs['dpt_pseudotime'].tolist(),
                "start_cluster": start_cluster,
                "dpt_groups": adata.obs['dpt_groups'].tolist() if 'dpt_groups' in adata.obs else None,
                "trajectory_genes": None
            }
        
        try:
            # Convert AnnData to Cell Data Set
            cds = Monocle3Wrapper.anndata_to_cell_data_set(adata)
            
            # Preprocess
            cds = Monocle3Wrapper.preprocess_cds(cds)
            
            # Reduce dimension
            cds = Monocle3Wrapper.reduce_dimension(cds)
            
            # Cluster cells
            cds = Monocle3Wrapper.cluster_cells(cds)
            
            # Learn graph
            cds = Monocle3Wrapper.learn_graph(cds)
            
            # Identify root cells if start_cluster is provided
            root_cells = None
            if start_cluster is not None:
                # Get cells from specified cluster
                partition_r = ro.r('partitions')(cds)
                with localconverter(ro.default_converter + pandas2ri.converter):
                    partitions = pandas2ri.rpy2py(partition_r)
                
                root_cells = adata.obs.index[partitions == int(start_cluster)].tolist()
                
                if len(root_cells) == 0:
                    logger.warning(f"No cells found in cluster {start_cluster}, using automatic root selection")
                    root_cells = None
            
            # Order cells
            cds = Monocle3Wrapper.order_cells(cds, root_cells=root_cells)
            
            # Get pseudotime
            pseudotime = Monocle3Wrapper.get_pseudotime(cds)
            
            # Get trajectory genes
            trajectory_genes = Monocle3Wrapper.find_trajectory_genes(cds)
            
            # Get partition information to determine start cluster
            partition_r = ro.r('partitions')(cds)
            with localconverter(ro.default_converter + pandas2ri.converter):
                partitions = pandas2ri.rpy2py(partition_r)
            
            # Map partition IDs to AnnData clusters
            if start_cluster is None and 'leiden' in adata.obs:
                # Get median pseudotime per cluster to identify start cluster
                adata.obs['monocle_pseudotime'] = pseudotime
                adata.obs['monocle_partition'] = partitions
                
                cluster_pt = adata.obs.groupby('leiden')['monocle_pseudotime'].median()
                start_cluster = cluster_pt.idxmin()
            
            # Create results dictionary
            results = {
                "pseudotime": pseudotime.tolist(),
                "start_cluster": start_cluster,
                "trajectory_genes": trajectory_genes.to_dict('records') if trajectory_genes is not None else None,
                "partitions": partitions.tolist()
            }
            
            # Add pseudotime to AnnData for future use
            adata.obs['monocle_pseudotime'] = pseudotime
            if trajectory_genes is not None:
                adata.uns['trajectory_genes'] = trajectory_genes.to_dict('records')
            
            return results
            
        except Exception as e:
            logger.error(f"Error running Monocle3 trajectory analysis: {str(e)}")
            raise