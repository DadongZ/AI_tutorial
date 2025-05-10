import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

# Global variable to store loaded data
adata = None

def get_adata():
    """Get the global adata object"""
    global adata
    return adata

def load_data(file_path: str) -> str:
    """Load single-cell data from a file."""
    global adata
    
    try:
        if file_path.endswith('.h5ad'):
            adata = sc.read_h5ad(file_path)
        elif file_path.endswith('.csv'):
            adata = sc.read_csv(file_path)
        elif os.path.isdir(file_path):
            adata = sc.read_10x_mtx(file_path)
        else:
            return f"Unsupported file format: {file_path}"
        
        return f"Successfully loaded data: {adata.n_obs} cells and {adata.n_vars} genes"
    except Exception as e:
        return f"Error loading data: {str(e)}"

def preprocess_data(min_genes: int = 5, min_cells: int = 3, normalize: bool = True) -> str:
    """Preprocess the loaded single-cell data."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Basic preprocessing
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        # For a small dataset, adjust PCA parameters
        # Get the number of highly variable genes
        n_hvgs = adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else adata.n_vars
        
        # Set n_pcs to a safe value (less than min of cells, genes and hvgs)
        n_pcs = min(15, min(adata.n_obs - 1, adata.n_vars - 1, n_hvgs - 1))
        print(f"Using {n_pcs} PCA components for a dataset with {adata.n_obs} cells and {adata.n_vars} genes")
        
        sc.pp.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        
        # Compute neighbors graph with reduced n_pcs
        n_neighbors = min(10, adata.n_obs - 1)  # Adjust neighbors based on cell count
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Run UMAP
        sc.tl.umap(adata)
        
        return f"Preprocessing complete. Remaining: {adata.n_obs} cells and {adata.n_vars} genes"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during preprocessing: {str(e)}\n{error_trace}"

def run_clustering(method: str = "louvain", resolution: float = 0.5) -> str:
    """Run clustering on the data."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Make sure neighbors have been computed
        if 'neighbors' not in adata.uns:
            # Compute neighbors if not done in preprocessing
            # Set n_pcs to a safe value (less than min of cells and genes)
            n_pcs = min(15, min(adata.n_obs - 1, adata.n_vars - 1))
            print(f"Computing neighbors with {n_pcs} PCs")
            
            n_neighbors = min(10, adata.n_obs - 1)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Try clustering methods with fallbacks
        cluster_key = None
        
        if method.lower() == "leiden":
            try:
                import leidenalg
                sc.tl.leiden(adata, resolution=resolution)
                cluster_key = 'leiden'
                print("Successfully ran Leiden clustering")
            except ImportError:
                print("Leiden algorithm not available. Falling back to Louvain.")
                method = "louvain"
        
        if method.lower() == "louvain":
            try:
                import louvain
                sc.tl.louvain(adata, resolution=resolution)
                cluster_key = 'louvain'
                print("Successfully ran Louvain clustering")
            except ImportError:
                print("Louvain package not available. Installing basic clustering.")
                # Fall back to basic k-means clustering
                from sklearn.cluster import KMeans
                n_clusters = max(2, min(20, adata.n_obs // 10))  # Reasonable number of clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
                cluster_key = 'kmeans'
                print(f"Used KMeans clustering with {n_clusters} clusters as fallback")
        
        if not cluster_key:
            return "Could not perform clustering. Please install either 'leidenalg' or 'python-louvain' package."
        
        cluster_counts = adata.obs[cluster_key].value_counts().to_dict()
        return f"{cluster_key.capitalize()} clustering complete. Found {len(cluster_counts)} clusters with distribution: {cluster_counts}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during clustering: {str(e)}\n{error_trace}"

def find_markers(group_by: str = None, n_genes: int = 10) -> str:
    """Find marker genes for clusters."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Find available clustering column
        if group_by is None:
            for col in ['louvain', 'leiden', 'kmeans']:
                if col in adata.obs.columns:
                    group_by = col
                    break
                    
            if group_by is None:
                return "No clustering found. Please run clustering first."
        
        if group_by not in adata.obs.columns:
            return f"Column '{group_by}' not found in data. Available columns: {list(adata.obs.columns)}"
        
        # Check for singleton clusters
        cluster_counts = adata.obs[group_by].value_counts()
        singleton_clusters = cluster_counts[cluster_counts == 1].index.tolist()
        
        if singleton_clusters:
            print(f"Warning: Clusters {', '.join(singleton_clusters)} have only one cell. Excluding them from marker gene analysis.")
            # Create a temporary column excluding singleton clusters
            adata.obs['_temp_group'] = adata.obs[group_by].copy()
            adata.obs.loc[adata.obs[group_by].isin(singleton_clusters), '_temp_group'] = 'singleton'
            
            # Use the temporary column for marker gene analysis
            sc.tl.rank_genes_groups(adata, '_temp_group', method='wilcoxon', groups=list(set(adata.obs['_temp_group']) - {'singleton'}))
            temp_group_key = '_temp_group'
        else:
            # Use the original grouping
            sc.tl.rank_genes_groups(adata, group_by, method='wilcoxon')
            temp_group_key = group_by
        
        # Extract results
        result = {}
        for group in set(adata.obs[temp_group_key]) - {'singleton'}:
            try:
                genes_df = sc.get.rank_genes_groups_df(adata, group=group)
                genes = genes_df['names'].tolist()[:n_genes]
                result[str(group)] = genes
            except KeyError:
                # This can happen if a group didn't yield results
                result[str(group)] = ["No significant markers found"]
        
        # Clean up temporary column if created
        if 'singleton' in set(adata.obs.get('_temp_group', [])):
            adata.obs = adata.obs.drop(columns=['_temp_group'])
        
        if result:
            return f"Found marker genes for each {group_by} (excluding singleton clusters): {result}"
        else:
            return f"No significant marker genes found for any cluster."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error finding markers: {str(e)}\n{error_trace}"

def run_monocle3_trajectory() -> str:
    """Run simplified trajectory analysis using scanpy's PAGA."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Check if neighbors have been computed
        if 'neighbors' not in adata.uns:
            return "Neighbors graph not computed. Please run preprocessing first."
        
        # Find available clustering column
        group_key = None
        for col in ['louvain', 'leiden', 'kmeans']:
            if col in adata.obs.columns:
                group_key = col
                break
                
        if group_key is None:
            return "No clustering found. Please run clustering first."
        
        # Run PAGA for trajectory inference
        sc.tl.paga(adata, groups=group_key)
        
        # Compute pseudotime using diffmap
        sc.tl.diffmap(adata)
        
        # Find root group 
        first_group = adata.obs[group_key].cat.categories[0] if hasattr(adata.obs[group_key], 'cat') else adata.obs[group_key].unique()[0]
        root_cell_idx = np.flatnonzero(adata.obs[group_key] == first_group)[0]
        adata.uns['iroot'] = root_cell_idx
        sc.tl.dpt(adata)
        
        # Store pseudotime in obs
        adata.obs['pseudotime'] = adata.obs['dpt_pseudotime']
        
        return "Trajectory analysis complete. Pseudotime values computed and stored."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during trajectory analysis: {str(e)}\n{error_trace}"

def plot_umap(color_by: str = None, save_path: str = "umap_plot.png") -> str:
    """Plot UMAP visualization of the data."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # If no color_by is specified, use available clustering
        if color_by is None or color_by not in adata.obs.columns:
            # First, see what's available
            available_clusterings = [col for col in ['louvain', 'leiden', 'kmeans'] 
                                   if col in adata.obs.columns]
            
            print(f"Available clustering columns: {available_clusterings}")
            
            if available_clusterings:
                color_by = available_clusterings[0]  # Use the first available
                print(f"Using {color_by} for coloring UMAP")
            elif 'condition' in adata.obs.columns:
                color_by = 'condition'
                print("Using condition for coloring UMAP")
            else:
                print("No suitable coloring found, using default visualization")
                color_by = None
        
        plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color=color_by, show=False)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return f"UMAP visualization saved to {save_path} colored by {color_by if color_by else 'default'}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error creating UMAP visualization: {str(e)}\n{error_trace}"

def get_data_summary() -> str:
    """Get a summary of the loaded data and analysis results."""
    global adata
    
    if adata is None:
        return "No data loaded."
    
    try:
        summary = {
            "cells": adata.n_obs,
            "genes": adata.n_vars,
            "available_obs": list(adata.obs.columns)
        }
        
        if hasattr(adata, 'obsm'):
            summary["available_obsm"] = list(adata.obsm.keys())
        
        # Check if clustering was performed
        if 'louvain' in adata.obs.columns:
            summary["clusters"] = adata.obs['louvain'].nunique()
        elif 'leiden' in adata.obs.columns:
            summary["clusters"] = adata.obs['leiden'].nunique()
        
        # Check if trajectory analysis was performed
        if 'pseudotime' in adata.obs.columns:
            summary["pseudotime_range"] = [float(adata.obs['pseudotime'].min()), 
                                           float(adata.obs['pseudotime'].max())]
        
        return f"Data summary: {summary}"
    except Exception as e:
        return f"Error getting data summary: {str(e)}"