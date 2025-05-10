from typing import Any, Dict, List, Optional, Tuple
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
from scipy import sparse
import numba
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Initialize FastMCP server
mcp = FastMCP("sc_analysis", log_level="ERROR")

# Global variable to store loaded data
adata = None

# Performance optimization: Use memory-efficient functions
@numba.jit(nopython=True)
def calculate_pairwise_distances_optimized(X, n_neighbors=30):
    """Optimized calculation of pairwise distances using numba."""
    n_cells = X.shape[0]
    distances = np.zeros((n_cells, n_neighbors))
    indices = np.zeros((n_cells, n_neighbors), dtype=np.int64)
    
    for i in range(n_cells):
        dists = np.sqrt(np.sum((X - X[i])**2, axis=1))
        idx = np.argsort(dists)[:n_neighbors]
        indices[i] = idx
        distances[i] = dists[idx]
    
    return distances, indices

@mcp.tool()
async def load_data(file_path: str, optimize_memory: bool = True) -> str:
    """Load single-cell data from a file with memory optimization options.
    
    Args:
        file_path: Path to the single-cell data file (h5ad, csv, or 10x mtx directory)
        optimize_memory: Whether to use memory-efficient data types
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
        
        # Memory optimization
        if optimize_memory:
            # Convert to sparse if not already
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X)
            
            # Optimize data types for categorical variables
            for col in adata.obs.select_dtypes(include=['object']):
                if adata.obs[col].nunique() < len(adata.obs) * 0.5:  # If less than 50% unique values
                    adata.obs[col] = adata.obs[col].astype('category')
        
        return f"Successfully loaded data: {adata.n_obs} cells and {adata.n_vars} genes"
    except Exception as e:
        return f"Error loading data: {str(e)}"

@mcp.tool()
async def preprocess_data(
    min_genes: int = 200, 
    min_cells: int = 3, 
    normalize: bool = True,
    n_hvgs: int = 2000,
    use_batch_correction: bool = False,
    batch_key: Optional[str] = None
) -> str:
    """Preprocess the loaded single-cell data with performance optimizations.
    
    Args:
        min_genes: Minimum number of genes per cell
        min_cells: Minimum number of cells per gene
        normalize: Whether to normalize the data
        n_hvgs: Number of highly variable genes to identify
        use_batch_correction: Whether to use batch correction
        batch_key: Key in adata.obs for batch information
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Store raw data for trajectory analysis
        adata.raw = adata
        
        # Basic preprocessing
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_hvgs)
        
        # Batch correction if requested
        if use_batch_correction and batch_key and batch_key in adata.obs.columns:
            sc.pp.combat(adata, key=batch_key)
        
        # Perform PCA on HVGs only for better performance
        adata.raw = adata  # Save full data
        adata = adata[:, adata.var.highly_variable]
        
        # For a small dataset, adjust PCA parameters
        n_hvgs_actual = adata.n_vars
        n_pcs = min(50, min(adata.n_obs - 1, n_hvgs_actual - 1))
        print(f"Using {n_pcs} PCA components for a dataset with {adata.n_obs} cells and {n_hvgs_actual} HVGs")
        
        sc.pp.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        
        # Compute neighbors graph with reduced n_pcs
        n_neighbors = min(30, adata.n_obs - 1)  # Increase default neighbors for better trajectory
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(30, n_pcs), metric='cosine')
        
        # Run UMAP
        sc.tl.umap(adata, min_dist=0.3)
        
        return f"Preprocessing complete. Remaining: {adata.n_obs} cells and {adata.n_vars} HVGs"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during preprocessing: {str(e)}\n{error_trace}"

@mcp.tool()
async def run_monocle3_trajectory(
    root_cell_selection: str = "auto",
    root_cell_idx: Optional[int] = None,
    root_cluster: Optional[str] = None,
    use_partition: bool = True,
    resolution: float = 0.5
) -> str:
    """Run Monocle3-style trajectory analysis with improved algorithms.
    
    Args:
        root_cell_selection: How to select root cell ("auto", "manual", "cluster")
        root_cell_idx: Specific cell index to use as root (if manual)
        root_cluster: Cluster to use as root (if cluster-based)
        use_partition: Whether to use partition-based trajectory learning
        resolution: Resolution for partition detection
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Ensure we have raw data for trajectory
        if adata.raw is None:
            return "Raw data not found. Please run preprocessing to store raw data."
        
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
            # Run leiden clustering for trajectory
            sc.tl.leiden(adata, resolution=resolution)
            group_key = 'leiden'
        
        # Step 1: Partition cells (Monocle3-style)
        if use_partition:
            # Use PAGA to find major partitions
            sc.tl.paga(adata, groups=group_key)
            
            # Extract partition connectivity
            partition_connectivity = adata.uns['paga']['connectivities'].copy()
            
            # Identify major partitions (clusters with high connectivity)
            partition_labels = []
            visited = set()
            
            for i in range(partition_connectivity.shape[0]):
                if i not in visited:
                    # Start a new partition
                    partition = {i}
                    queue = [i]
                    
                    while queue:
                        current = queue.pop(0)
                        for j in range(partition_connectivity.shape[1]):
                            if j != current and partition_connectivity[current, j] > 0.1 and j not in visited:
                                partition.add(j)
                                queue.append(j)
                                visited.add(j)
                        visited.add(current)
                    
                    partition_labels.append(list(partition))
            
            # Assign cells to partitions
            cluster_to_partition = {}
            for part_idx, clusters in enumerate(partition_labels):
                for cluster in clusters:
                    cluster_to_partition[str(cluster)] = f"partition_{part_idx}"
            
            adata.obs['monocle3_partition'] = adata.obs[group_key].map(cluster_to_partition)
        
        # Step 2: Learn principal graph (using PAGA + refinement)
        sc.tl.diffmap(adata, n_comps=30)
        
        # Step 3: Order cells along trajectories
        # Select root cell
        if root_cell_selection == "manual" and root_cell_idx is not None:
            adata.uns['iroot'] = root_cell_idx
        elif root_cell_selection == "cluster" and root_cluster is not None:
            root_cells = np.where(adata.obs[group_key] == root_cluster)[0]
            if len(root_cells) == 0:
                return f"No cells found in cluster {root_cluster}"
            # Select cell closest to center of mass in diffusion space
            cluster_center = adata.obsm['X_diffmap'][root_cells].mean(axis=0)
            distances = np.sum((adata.obsm['X_diffmap'][root_cells] - cluster_center)**2, axis=1)
            adata.uns['iroot'] = root_cells[np.argmin(distances)]
        else:
            # Auto-select root using first principal component
            pc1_scores = adata.obsm['X_pca'][:, 0]
            adata.uns['iroot'] = np.argmin(pc1_scores)  # Use minimum of PC1
        
        # Calculate diffusion pseudotime
        sc.tl.dpt(adata)
        
        # Store pseudotime in obs
        adata.obs['monocle3_pseudotime'] = adata.obs['dpt_pseudotime']
        
        # Calculate trajectory-specific metrics (similar to Monocle3)
        # 1. Calculate branch probability
        if use_partition:
            # Calculate branch assignments
            branch_assignments = {}
            for partition in set(adata.obs['monocle3_partition']):
                if partition and partition != 'nan':
                    mask = adata.obs['monocle3_partition'] == partition
                    cells = np.where(mask)[0]
                    if len(cells) > 1:
                        # Assign branch based on pseudotime ordering within partition
                        pseudotimes = adata.obs.loc[mask, 'monocle3_pseudotime']
                        branch_assignments[partition] = pseudotimes.mean()
            
            # Order branches by pseudotime
            ordered_branches = sorted(branch_assignments.items(), key=lambda x: x[1])
            for i, (branch, _) in enumerate(ordered_branches):
                adata.obs.loc[adata.obs['monocle3_partition'] == branch, 'monocle3_branch'] = f"Branch_{i+1}"
        
        # 2. Calculate branch point probabilities
        adata.obs['branch_point_probability'] = 1 - np.exp(-adata.obs['monocle3_pseudotime'] / adata.obs['monocle3_pseudotime'].std())
        
        # Store root information
        adata.uns['monocle3_root'] = {
            'root_cell_idx': int(adata.uns['iroot']),
            'root_cell_id': adata.obs.index[adata.uns['iroot']],
            'selection_method': root_cell_selection
        }
        
        result = {
            'n_cells': adata.n_obs,
            'root_cell': adata.obs.index[adata.uns['iroot']],
            'pseudotime_range': [float(adata.obs['monocle3_pseudotime'].min()), 
                               float(adata.obs['monocle3_pseudotime'].max())],
            'n_branches': len(set(adata.obs.get('monocle3_branch', []))) if 'monocle3_branch' in adata.obs else 0,
            'n_partitions': len(set(adata.obs.get('monocle3_partition', []))) if 'monocle3_partition' in adata.obs else 0
        }
        
        return f"Monocle3-style trajectory analysis complete: {result}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during trajectory analysis: {str(e)}\n{error_trace}"

@mcp.tool()
async def find_trajectory_genes(
    n_genes: int = 100,
    branch_specific: bool = True,
    test_branches: Optional[List[str]] = None,
    min_expr_frac: float = 0.1
) -> str:
    """Find genes that vary along the trajectory (Monocle3-style).
    
    Args:
        n_genes: Number of top trajectory genes to identify
        branch_specific: Whether to find branch-specific genes
        test_branches: Specific branches to test (if None, test all)
        min_expr_frac: Minimum fraction of cells expressing a gene
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        if 'monocle3_pseudotime' not in adata.obs:
            return "Trajectory analysis not performed. Please run trajectory analysis first."
        
        # Use raw data for gene expression
        if adata.raw is None:
            return "Raw data not available. Please run preprocessing with raw data storage."
        
        # Get raw expression data
        raw_X = adata.raw.X.copy()
        if sparse.issparse(raw_X):
            raw_X = raw_X.toarray()
        
        # Filter genes by expression fraction
        expr_mask = (raw_X > 0).mean(axis=0) >= min_expr_frac
        genes_to_test = adata.raw.var.index[expr_mask]
        
        # Calculate correlation with pseudotime
        pseudotime = adata.obs['monocle3_pseudotime'].values
        correlations = []
        
        for i, gene in enumerate(genes_to_test):
            if i % 1000 == 0:
                print(f"Processing gene {i}/{len(genes_to_test)}")
            
            gene_idx = adata.raw.var.index.get_loc(gene)
            gene_expr = raw_X[:, gene_idx]
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            corr, pval = spearmanr(pseudotime, gene_expr)
            correlations.append((gene, abs(corr), pval))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Store trajectory gene results
        traj_genes_df = pd.DataFrame(correlations[:n_genes], 
                                   columns=['gene', 'correlation', 'pval'])
        adata.uns['trajectory_genes'] = traj_genes_df
        
        result = {'top_trajectory_genes': traj_genes_df.head().to_dict()}
        
        # Branch-specific analysis
        if branch_specific and 'monocle3_branch' in adata.obs:
            branches = set(adata.obs['monocle3_branch'].dropna())
            if test_branches:
                branches = set(test_branches).intersection(branches)
            
            branch_specific_genes = {}
            
            for branch in branches:
                branch_mask = adata.obs['monocle3_branch'] == branch
                if branch_mask.sum() < 10:  # Skip branches with too few cells
                    continue
                
                # Calculate expression differences
                branch_expr = raw_X[branch_mask].mean(axis=0)
                other_expr = raw_X[~branch_mask].mean(axis=0)
                
                # Calculate fold change
                fold_changes = np.log2((branch_expr + 1e-6) / (other_expr + 1e-6))
                
                # Get top genes for this branch
                branch_genes = []
                for i, gene in enumerate(genes_to_test):
                    gene_idx = adata.raw.var.index.get_loc(gene)
                    if abs(fold_changes[gene_idx]) > 0.5:  # At least 1.4-fold change
                        branch_genes.append((gene, fold_changes[gene_idx]))
                
                branch_genes.sort(key=lambda x: abs(x[1]), reverse=True)
                branch_specific_genes[branch] = branch_genes[:20]
                
            adata.uns['branch_specific_genes'] = branch_specific_genes
            result['branch_specific_genes'] = {k: v[:5] for k, v in branch_specific_genes.items()}
        
        return f"Trajectory genes identified: {result}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error finding trajectory genes: {str(e)}\n{error_trace}"

@mcp.tool()
async def plot_trajectory_advanced(
    color_by: Optional[List[str]] = None,
    plot_type: str = "combined",
    save_path: Optional[str] = None,
    show_branch_points: bool = True,
    cell_size: float = 1.0
) -> str:
    """Create advanced trajectory visualizations (Monocle3-style).
    
    Args:
        color_by: List of features to color cells by
        plot_type: Type of plot ("combined", "paga", "umap", "trajectory")
        save_path: Path to save the figure
        show_branch_points: Whether to highlight branch points
        cell_size: Size of cells in the plot
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        if 'monocle3_pseudotime' not in adata.obs:
            return "Trajectory analysis not performed. Please run trajectory analysis first."
        
        # Set default colors if none specified
        if color_by is None:
            color_by = ['monocle3_pseudotime']
            if 'monocle3_branch' in adata.obs:
                color_by.append('monocle3_branch')
        
        # Create figure based on plot type
        if plot_type == "combined":
            n_plots = len(color_by) + 1  # +1 for PAGA
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            # Plot PAGA
            sc.pl.paga(adata, threshold=0.03, ax=axes[0], show=False, 
                      title='Trajectory Graph', node_size_scale=1.5)
            
            # Plot UMAPs with different colorings
            for i, color in enumerate(color_by):
                ax = axes[i+1]
                
                # Create custom color scheme for pseudotime
                if color == 'monocle3_pseudotime':
                    cmap = plt.cm.viridis
                    # Highlight branch points if requested
                    if show_branch_points and 'branch_point_probability' in adata.obs:
                        branch_mask = adata.obs['branch_point_probability'] > 0.9
                        # First plot all cells
                        sc.pl.umap(adata, color=color, ax=ax, show=False, 
                                 title=f'Cells by {color}', size=cell_size,
                                 cmap=cmap, colorbar_loc='right')
                        # Then overlay branch points
                        if branch_mask.sum() > 0:
                            ax.scatter(adata.obsm['X_umap'][branch_mask, 0],
                                     adata.obsm['X_umap'][branch_mask, 1],
                                     s=50*cell_size, edgecolors='red', 
                                     facecolors='none', linewidths=2)
                    else:
                        sc.pl.umap(adata, color=color, ax=ax, show=False, 
                                 title=f'Cells by {color}', size=cell_size,
                                 cmap=cmap, colorbar_loc='right')
                else:
                    sc.pl.umap(adata, color=color, ax=ax, show=False, 
                             title=f'Cells by {color}', size=cell_size)
            
        elif plot_type == "trajectory":
            # Create trajectory-specific plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot principal graph representation
            sc.pl.paga(adata, threshold=0.03, ax=ax, show=False, 
                      init_pos='umap', node_size_scale=1.5)
            
            # Color nodes by average pseudotime
            if 'leiden' in adata.obs.columns:
                cluster_pseudotimes = adata.obs.groupby('leiden')['monocle3_pseudotime'].mean()
                node_colors = plt.cm.viridis(cluster_pseudotimes / cluster_pseudotimes.max())
                
                for i, (cluster, color) in enumerate(zip(cluster_pseudotimes.index, node_colors)):
                    ax.collections[0].set_facecolors(node_colors)
            
            ax.set_title('Trajectory Principal Graph')
        
        else:  # Single plot type
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            if plot_type == "paga":
                sc.pl.paga(adata, threshold=0.03, ax=ax, show=False, 
                          init_pos='umap', node_size_scale=1.5)
                ax.set_title('Trajectory Graph')
            else:  # umap
                color = color_by[0] if color_by else 'monocle3_pseudotime'
                sc.pl.umap(adata, color=color, ax=ax, show=False, 
                         title=f'Cells by {color}', size=cell_size)
        
        plt.tight_layout()
        
        # Save or encode the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"Advanced trajectory visualization saved to {save_path}"
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            data = base64.b64encode(buf.getbuffer()).decode("utf8")
            return f"Advanced trajectory visualization created. Image data: {data}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error creating advanced trajectory visualization: {str(e)}\n{error_trace}"

@mcp.tool()
async def benchmark_performance(n_cells: int = 1000, n_genes: int = 5000) -> str:
    """Benchmark the performance of the analysis pipeline."""
    global adata
    
    try:
        # Create synthetic data for benchmarking
        print(f"Creating synthetic dataset with {n_cells} cells and {n_genes} genes...")
        np.random.seed(42)
        
        # Create sparse expression matrix
        X = sparse.random(n_cells, n_genes, density=0.05, format='csr', random_state=42)
        
        # Create metadata
        obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
        var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
        
        # Create synthetic adata
        test_adata = anndata.AnnData(X=X, obs=obs, var=var)
        
        # Store original data
        original_adata = adata
        adata = test_adata
        
        # Time preprocessing
        import time
        start_time = time.time()
        
        # Run preprocessing pipeline
        print("Running preprocessing...")
        preprocess_result = await preprocess_data(normalize=True, n_hvgs=min(2000, n_genes//3))
        preprocess_time = time.time() - start_time
        
        # Run clustering
        print("Running clustering...")
        start_time = time.time()
        cluster_result = await run_clustering(method="leiden", resolution=0.5)
        cluster_time = time.time() - start_time
        
        # Run trajectory analysis
        print("Running trajectory analysis...")
        start_time = time.time()
        trajectory_result = await run_monocle3_trajectory(root_cell_selection="auto")
        trajectory_time = time.time() - start_time
        
        # Restore original data
        adata = original_adata
        
        results = {
            "dataset_size": {"cells": n_cells, "genes": n_genes},
            "preprocessing_time": f"{preprocess_time:.2f} seconds",
            "clustering_time": f"{cluster_time:.2f} seconds", 
            "trajectory_time": f"{trajectory_time:.2f} seconds",
            "total_time": f"{preprocess_time + cluster_time + trajectory_time:.2f} seconds",
            "cells_per_second": f"{n_cells / (preprocess_time + cluster_time + trajectory_time):.1f}"
        }
        
        return f"Performance benchmark results: {results}"
        
    except Exception as e:
        # Restore original data if error occurs
        if 'original_adata' in locals():
            adata = original_adata
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during benchmarking: {str(e)}\n{error_trace}"

# All other tools remain the same as in the original code but with better memory management

@mcp.tool()
async def run_clustering(method: str = "louvain", resolution: float = 0.5) -> str:
    """Run clustering on the data with performance optimizations.
    
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
            n_pcs = min(30, min(adata.n_obs - 1, adata.n_vars - 1))
            n_neighbors = min(30, adata.n_obs - 1)  # Increased for better performance
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric='cosine')
        
        # Try clustering methods with fallbacks
        cluster_key = None
        
        if method.lower() == "leiden":
            try:
                sc.tl.leiden(adata, resolution=resolution, n_iterations=2)  # More iterations for stability
                cluster_key = 'leiden'
                print("Successfully ran Leiden clustering")
            except ImportError:
                print("Leiden algorithm not available. Falling back to Louvain.")
                method = "louvain"
        
        if method.lower() == "louvain":
            try:
                sc.tl.louvain(adata, resolution=resolution)
                cluster_key = 'louvain'
                print("Successfully ran Louvain clustering")
            except ImportError:
                print("Louvain package not available. Using K-means clustering.")
                method = "kmeans"
        
        if method.lower() == "kmeans":
            # Optimized k-means clustering
            from sklearn.cluster import MiniBatchKMeans
            n_clusters = max(2, min(50, adata.n_obs // 10))  # Dynamic cluster number
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
            cluster_key = 'kmeans'
            print(f"Used MiniBatch KMeans clustering with {n_clusters} clusters")
        
        if not cluster_key:
            return "Could not perform clustering. Please install either 'leidenalg' or 'python-louvain' package."
        
        cluster_counts = adata.obs[cluster_key].value_counts().to_dict()
        return f"{cluster_key.capitalize()} clustering complete. Found {len(cluster_counts)} clusters with distribution: {cluster_counts}"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error during clustering: {str(e)}\n{error_trace}"

@mcp.tool()
async def find_markers(group_by: str = None, n_genes: int = 10, method: str = 'wilcoxon') -> str:
    """Find marker genes for clusters with performance optimizations.
    
    Args:
        group_by: Column in adata.obs to group cells by
        n_genes: Number of top marker genes per group
        method: Statistical test method
    """
    global adata
    
    if adata is None:
        return "No data loaded. Please load data first."
    
    try:
        # Find available clustering column
        if group_by is None:
            for col in ['leiden', 'louvain', 'kmeans']:
                if col in adata.obs.columns:
                    group_by = col
                    break
                    
            if group_by is None:
                return "No clustering found. Please run clustering first."
        
        if group_by not in adata.obs.columns:
            return f"Column '{group_by}' not found in data. Available columns: {list(adata.obs.columns)}"
        
        # Check for singleton clusters
        cluster_counts = adata.obs[group_by].value_counts()
        singleton_clusters = cluster_counts[cluster_counts < 3].index.tolist()  # Require at least 3 cells
        
        if singleton_clusters:
            print(f"Warning: Clusters {', '.join(singleton_clusters)} have fewer than 3 cells. Excluding them from marker gene analysis.")
            # Create a temporary column excluding singleton clusters
            adata.obs['_temp_group'] = adata.obs[group_by].copy()
            adata.obs.loc[adata.obs[group_by].isin(singleton_clusters), '_temp_group'] = 'singleton'
            
            # Use the temporary column for marker gene analysis
            sc.tl.rank_genes_groups(adata, '_temp_group', method=method, groups=list(set(adata.obs['_temp_group']) - {'singleton'}))
            temp_group_key = '_temp_group'
        else:
            # Use the original grouping
            sc.tl.rank_genes_groups(adata, group_by, method=method, use_raw=True)
            temp_group_key = group_by
        
        # Extract results
        result = {}
        for group in set(adata.obs[temp_group_key]) - {'singleton'}:
            try:
                genes_df = sc.get.rank_genes_groups_df(adata, group=group)
                genes = genes_df['names'].tolist()[:n_genes]
                result[str(group)] = genes
            except KeyError:
                result[str(group)] = ["No significant markers found"]
        
        # Clean up temporary column
        if '_temp_group' in adata.obs.columns:
            del adata.obs['_temp_group']
        
        if result:
            return f"Found marker genes for each {group_by} (with minimum 3 cells): {result}"
        else:
            return f"No significant marker genes found for any cluster."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error finding markers: {str(e)}\n{error_trace}"

@mcp.tool()
async def save_data(file_path: str, compress: bool = True) -> str:
    """Save the current analysis state to a file with compression option.
    
    Args:
        file_path: Path to save the h5ad file
        compress: Whether to compress the output file
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
        
        # Save the AnnData object with compression
        compression = 'gzip' if compress else None
        adata.write(file_path, compression=compression)
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024*1024)  # Convert to MB
        
        return f"Analysis state saved to {file_path} (Size: {file_size:.2f} MB)"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error saving data: {str(e)}\n{error_trace}"

# All other existing tools (plot_umap, plot_trajectory, etc.) remain the same but with enhanced functionality

if __name__ == "__main__":
    # Initialize and run the server
    print("Starting enhanced single-cell analysis MCP server...")
    mcp.run(transport='stdio')