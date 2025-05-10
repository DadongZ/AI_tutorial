import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

# Import your functions from test_monocle.py
from test_monocle import (
    load_data, preprocess_data, run_clustering, 
    find_markers, run_monocle3_trajectory, 
    plot_umap, get_data_summary, get_adata
)

def create_test_dataset():
    """Create a small test dataset and save it as h5ad"""
    print("Creating test dataset...")
    
    # Load built-in dataset
    temp_adata = sc.datasets.pbmc3k()
    
    # Subsample to make it faster for testing
    temp_adata = temp_adata[:200, :500].copy()  # Smaller sample for faster testing
    
    # Add a mock condition column
    np.random.seed(42)
    temp_adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=temp_adata.n_obs, p=[0.6, 0.4])
    
    # Create output directory if it doesn't exist
    os.makedirs("./output", exist_ok=True)
    
    # Save the h5ad file
    test_file = './output/test_data.h5ad'
    temp_adata.write(test_file)
    print(f"Created test dataset with {temp_adata.n_obs} cells and {temp_adata.n_vars} genes")
    
    return test_file

def run_test():
    """Run a test of the single-cell analysis functions"""
    # Create output directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files in output directory
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path) and file != "test_data.h5ad":
            os.unlink(file_path)
    
    # Create test dataset
    test_file = create_test_dataset()
    
    # Test data loading
    print("\n=== Testing data loading ===")
    result = load_data(test_file)
    print(result)
    
    # Get the adata object
    adata = get_adata()
    if adata is None:
        print("ERROR: adata is None after loading. Check test_monocle.py")
        return
    
    # Test preprocessing
    print("\n=== Testing preprocessing ===")
    result = preprocess_data(min_genes=5, min_cells=3)
    print(result)
    
    # Get the updated adata
    adata = get_adata()
    if adata is None:
        print("ERROR: adata is None after preprocessing. Check test_monocle.py")
        return
    
    # Create a basic visualization after preprocessing
    try:
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        sc.pl.highest_expr_genes(adata, n_top=20, show=False)
        plt.title("Top expressed genes")
        
        plt.subplot(2, 2, 2)
        if 'highly_variable' in adata.var:
            sc.pl.highly_variable_genes(adata, show=False)
            plt.title("Highly variable genes")
        
        plt.subplot(2, 2, 3)
        sc.pl.pca_variance_ratio(adata, n_pcs=15, show=False)
        plt.title("PCA variance ratio")
        
        plt.subplot(2, 2, 4)
        sc.pl.pca(adata, color='condition', show=False)
        plt.title("PCA colored by condition")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "preprocessing_plots.png"), dpi=300)
        plt.close()
        print(f"Saved preprocessing visualizations to {os.path.join(output_dir, 'preprocessing_plots.png')}")
    except Exception as e:
        print(f"Error creating preprocessing visualizations: {str(e)}")
    
    # Test clustering
    print("\n=== Testing clustering ===")
    result = run_clustering(method="louvain")
    print(result)
    
    # Get the updated adata again
    adata = get_adata()
    
    # Test marker gene finding
    print("\n=== Testing marker gene finding ===")
    result = find_markers()
    print(result)
    
    # Update adata again
    adata = get_adata()
    
    # Create markers visualization
    # Find which clustering was used
    cluster_col = None
    for col in ['louvain', 'leiden', 'kmeans']:
        if col in adata.obs.columns:
            cluster_col = col
            break
    
# In the marker gene visualization section
    if cluster_col and 'rank_genes_groups' in adata.uns:
        try:
            plt.figure(figsize=(12, 10))
            
            # Create simple dotplot and heatmap that don't rely on the exact groupby
            sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, standard_scale='var', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "marker_genes_dotplot.png"), dpi=300)
            plt.close()
            
            plt.figure(figsize=(12, 10))
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, standard_scale='var', show_gene_labels=True, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "marker_genes_heatmap.png"), dpi=300)
            plt.close()
            
            print(f"Saved marker gene visualizations to {output_dir}")
        except Exception as e:
            print(f"Could not create marker gene visualizations: {str(e)}")
    
    # Test trajectory analysis
    print("\n=== Testing trajectory analysis ===")
    result = run_monocle3_trajectory()
    print(result)
    
    # Update adata again
    adata = get_adata()
    
    # Create trajectory visualization
    if 'paga' in adata.uns and 'pseudotime' in adata.obs:
        try:
            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            sc.pl.paga(adata, threshold=0.03, show=False)
            plt.title("PAGA Graph")
            
            plt.subplot(1, 2, 2)
            sc.pl.umap(adata, color='pseudotime', show=False)
            plt.title("Pseudotime")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "trajectory_visualization.png"), dpi=300)
            plt.close()
            print(f"Saved trajectory visualization to {os.path.join(output_dir, 'trajectory_visualization.png')}")
        except Exception as e:
            print(f"Error creating trajectory visualization: {str(e)}")
    
    # Test visualization
    print("\n=== Testing visualization ===")
    result = plot_umap(color_by=cluster_col, save_path=os.path.join(output_dir, "umap_plot.png"))
    print(result)
    
    # Update adata one more time
    adata = get_adata()
    
    # Create combined UMAP visualization
    try:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        sc.pl.umap(adata, color=cluster_col, show=False)
        plt.title(f"UMAP by {cluster_col}")
        
        plt.subplot(1, 3, 2)
        sc.pl.umap(adata, color='condition', show=False)
        plt.title("UMAP by condition")
        
        plt.subplot(1, 3, 3)
        sc.pl.umap(adata, color='pseudotime', show=False)
        plt.title("UMAP by pseudotime")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_umap.png"), dpi=300)
        plt.close()
        print(f"Saved combined UMAP visualization to {os.path.join(output_dir, 'combined_umap.png')}")
    except Exception as e:
        print(f"Error creating combined UMAP visualization: {str(e)}")
    
    # Test summary
    print("\n=== Testing data summary ===")
    result = get_data_summary()
    print(result)
    
    # Save the final AnnData object for further inspection
    adata.write(os.path.join(output_dir, "final_analysis.h5ad"))
    print(f"Saved final analysis data to {os.path.join(output_dir, 'final_analysis.h5ad')}")
    
    print("\nAll tests completed! Results saved to the './output' directory")

if __name__ == "__main__":
    run_test()