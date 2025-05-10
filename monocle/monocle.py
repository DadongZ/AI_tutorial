from typing import Any, Dict, List, Optional
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
import asyncio
import os
import base64
from io import BytesIO

# Initialize FastMCP server
mcp = FastMCP("sc_analysis", log_level="ERROR")

# Global variable to store loaded data
adata = None

@mcp.tool()
async def load_data(file_path: str) -> str:
    """Load single-cell data from a file.
    
    Args:
        file_path: Path to the single-cell data file (h5ad, csv, or 10x mtx directory)
    """
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

@mcp.tool()
async def preprocess_data(min_genes: int = 200, min_cells: int = 3, normalize: bool = True) -> str:
    """Preprocess the loaded single-cell data.
    
    Args:
        min_genes: Minimum number of genes per cell
        min_cells: Minimum number of cells per gene
        normalize: Whether to normalize the data
    """
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
        n_hvgs = adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else adata.n_vars
        
        # Set n_pcs to a safe value (less than min of cells, genes and hvgs)
        n_pcs = min(50, min(adata.n_obs - 1, adata.n_vars - 1, n_hvgs - 1))
        print(f"Using {n_pcs} PCA components for a dataset with {adata.n_obs} cells and {adata.n_vars} genes")
        
        sc.pp.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        
        # Compute neighbors graph with reduced n_pcs
        n_neighbors = min(15, adata.n_obs - 1)  # Adjust neighbors based on cell count
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs))
        
        # Run UMAP
        sc.tl.umap(adata)
        
        return f"Preprocessing complete. Remaining: {adata.n_obs} cells and {adata.n_vars} genes"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during preprocessing: {str(e)}\n{error_trace}"

@mcp.tool()
async def run_clustering(method: str = "louvain", resolution: float = 0.5) -> str:
    """Run clustering on the data.
    
    Args:
        method: Clustering method ('louvain', 'leiden', or 'kmeans')
        resolution: Resolution parameter for community detection methods
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Make sure neighbors have been computed
        if 'neighbors' not in adata.uns:
            # Compute neighbors if not done in preprocessing
            # Set n_pcs to a safe value (less than min of cells and genes)
            n_pcs = min(30, min(adata.n_obs - 1, adata.n_vars - 1))
            print(f"Computing neighbors with {n_pcs} PCs")
            
            n_neighbors = min(15, adata.n_obs - 1)
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
                print("Louvain package not available. Using K-means clustering.")
                method = "kmeans"
        
        if method.lower() == "kmeans":
            # Fall back to basic k-means clustering
            from sklearn.cluster import KMeans
            n_clusters = max(2, min(20, adata.n_obs // 10))  # Reasonable number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
            cluster_key = 'kmeans'
            print(f"Used KMeans clustering with {n_clusters} clusters")
        
        if not cluster_key:
            return "Could not perform clustering. Please install either 'leidenalg' or 'python-louvain' package."
        
        cluster_counts = adata.obs[cluster_key].value_counts().to_dict()
        return f"{cluster_key.capitalize()} clustering complete. Found {len(cluster_counts)} clusters with distribution: {cluster_counts}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during clustering: {str(e)}\n{error_trace}"

@mcp.tool()
async def find_markers(group_by: str = None, n_genes: int = 10) -> str:
    """Find marker genes for clusters.
    
    Args:
        group_by: Column in adata.obs to group cells by (default: auto-detect clustering)
        n_genes: Number of top marker genes to return per group
    """
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
        
        # DO NOT clean up temporary column - needed for visualization
        
        if result:
            return f"Found marker genes for each {group_by} (excluding singleton clusters): {result}"
        else:
            return f"No significant marker genes found for any cluster."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error finding markers: {str(e)}\n{error_trace}"

@mcp.tool()
async def run_monocle3_trajectory() -> str:
    """Run simplified trajectory analysis using scanpy's PAGA and diffusion pseudotime.
    
    This provides functionality similar to Monocle3's trajectory analysis.
    """
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

@mcp.tool()
async def plot_umap(color_by: str = None, save_path: Optional[str] = None) -> str:
    """Plot UMAP visualization of the data.
    
    Args:
        color_by: Column in adata.obs to color cells by (default: auto-detect)
        save_path: Path to save the figure (if None, returns a base64 encoded image)
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # If no color_by is specified, use available clustering
        if color_by is None:
            for col in ['louvain', 'leiden', 'kmeans']:
                if col in adata.obs.columns:
                    color_by = col
                    break
            
            if color_by is None and 'condition' in adata.obs.columns:
                color_by = 'condition'
        
        if color_by is not None and color_by not in adata.obs.columns:
            return f"Column '{color_by}' not found in data. Available columns: {list(adata.obs.columns)}"
        
        plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color=color_by, show=False)
        
        # Save or encode the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"UMAP visualization saved to {save_path}"
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            data = base64.b64encode(buf.getbuffer()).decode("utf8")
            return f"UMAP visualization created. Image data: {data}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error creating UMAP visualization: {str(e)}\n{error_trace}"

@mcp.tool()
async def plot_trajectory(save_path: Optional[str] = None) -> str:
    """Plot trajectory analysis results.
    
    Args:
        save_path: Path to save the figure (if None, returns a base64 encoded image)
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Check required data
        if 'paga' not in adata.uns:
            return "Trajectory analysis not performed. Please run trajectory analysis first."
        
        if 'pseudotime' not in adata.obs:
            return "Pseudotime values not found. Please run trajectory analysis first."
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot PAGA graph
        sc.pl.paga(adata, threshold=0.03, ax=axes[0], show=False)
        axes[0].set_title('PAGA Graph')
        
        # Plot force-directed layout with pseudotime
        sc.pl.umap(adata, color='pseudotime', ax=axes[1], show=False)
        axes[1].set_title('Pseudotime')
        
        # Save or encode the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"Trajectory visualization saved to {save_path}"
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            data = base64.b64encode(buf.getbuffer()).decode("utf8")
            return f"Trajectory visualization created. Image data: {data}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error creating trajectory visualization: {str(e)}\n{error_trace}"

@mcp.tool()
async def plot_marker_genes(n_genes: int = 5, save_path: Optional[str] = None) -> str:
    """Plot heatmap and dotplot of marker genes.
    
    Args:
        n_genes: Number of top genes per group to include
        save_path: Path to save the figure (if None, returns a base64 encoded image)
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        if 'rank_genes_groups' not in adata.uns:
            return "Marker genes not computed. Please run find_markers first."
        
        # Create a combined figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot dotplot
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=n_genes, ax=axes[0], show=False)
        axes[0].set_title('Marker Genes Dotplot')
        
        # Plot heatmap with more genes
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=n_genes, ax=axes[1], 
                                       show_gene_labels=True, show=False)
        axes[1].set_title('Marker Genes Heatmap')
        
        plt.tight_layout()
        
        # Save or encode the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"Marker genes visualization saved to {save_path}"
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            data = base64.b64encode(buf.getbuffer()).decode("utf8")
            return f"Marker genes visualization created. Image data: {data}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error creating marker genes visualization: {str(e)}\n{error_trace}"

@mcp.tool()
async def get_data_summary() -> str:
    """Get a summary of the loaded data and analysis results."""
    global adata
    
    if adata is None:
        return "No data loaded."
    
    try:
        summary = {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "available_obs": list(adata.obs.columns)
        }
        
        if hasattr(adata, 'obsm'):
            summary["available_obsm"] = list(adata.obsm.keys())
        
        if hasattr(adata, 'uns'):
            summary["available_analyses"] = []
            if 'neighbors' in adata.uns:
                summary["available_analyses"].append("neighbors")
            if 'pca' in adata.uns:
                summary["available_analyses"].append("pca")
            if 'rank_genes_groups' in adata.uns:
                summary["available_analyses"].append("marker_genes")
            if 'paga' in adata.uns:
                summary["available_analyses"].append("trajectory")
        
        # Check if clustering was performed
        for cluster_method in ['louvain', 'leiden', 'kmeans']:
            if cluster_method in adata.obs.columns:
                summary["clustering_method"] = cluster_method
                summary["clusters"] = int(adata.obs[cluster_method].nunique())
                break
        
        # Check if trajectory analysis was performed
        if 'pseudotime' in adata.obs.columns:
            summary["pseudotime_range"] = [float(adata.obs['pseudotime'].min()), 
                                           float(adata.obs['pseudotime'].max())]
        
        return f"Data summary: {summary}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error getting data summary: {str(e)}\n{error_trace}"

@mcp.tool()
async def save_data(file_path: str) -> str:
    """Save the current analysis state to a file.
    
    Args:
        file_path: Path to save the h5ad file
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Ensure file has correct extension
        if not file_path.endswith('.h5ad'):
            file_path += '.h5ad'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save the AnnData object
        adata.write(file_path)
        return f"Analysis state saved to {file_path}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error saving data: {str(e)}\n{error_trace}"

@mcp.tool()
async def get_available_genes() -> str:
    """Get a list of available genes in the dataset."""
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        if adata.n_vars > 100:
            # If there are many genes, just return the count and a sample
            sample_genes = list(adata.var_names[:20])
            return f"Dataset contains {adata.n_vars} genes. Sample of genes: {sample_genes}"
        else:
            # If there are few genes, return all of them
            all_genes = list(adata.var_names)
            return f"Dataset contains {adata.n_vars} genes: {all_genes}"
    except Exception as e:
        return f"Error getting gene list: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    print("Starting single-cell analysis MCP server...")
    mcp.run(transport='stdio')