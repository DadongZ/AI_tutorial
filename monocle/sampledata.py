import scanpy as sc
import numpy as np

# Load built-in dataset
adata = sc.datasets.pbmc3k()

# Add a mock condition column
np.random.seed(42)
adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=adata.n_obs, p=[0.6, 0.4])

# Basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Save the h5ad file
adata.write('test_data.h5ad')
print(f"Created test dataset with {adata.n_obs} cells and {adata.n_vars} genes")