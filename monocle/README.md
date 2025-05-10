# Single-Cell Analysis MCP Server

A Model Context Protocol (MCP) server for comprehensive single-cell RNA sequencing analysis using Python and scanpy, with Monocle3-style trajectory analysis and performance optimizations for large datasets.

## Features

### Core Single-Cell Analysis
- **Data Loading**: Support for h5ad, csv, and 10x mtx formats with memory optimization
- **Preprocessing**: Quality control, normalization, batch correction, and dimensionality reduction
- **Clustering**: Multiple methods (Leiden, Louvain, K-means) with automatic fallbacks
- **Marker Gene Discovery**: Differential expression analysis with statistical testing
- **UMAP/PCA Visualization**: Interactive plotting with customizable color schemes

### Advanced Trajectory Analysis (Monocle3-style)
- **Partition-based Learning**: Automatic cell partition detection for complex trajectories
- **Principal Graph Construction**: PAGA-based graph inference with diffusion pseudotime
- **Branch Detection**: Automatic identification of trajectory branches and endpoints
- **Root Cell Selection**: Manual, cluster-based, or automatic root selection methods
- **Trajectory Genes**: Identification of pseudotime-varying and branch-specific genes
- **Branch Point Analysis**: Calculation of branch point probabilities

### Performance Optimizations
- **Memory Management**: Sparse matrix operations and categorical data optimization
- **Batch Processing**: MiniBatch algorithms for large dataset compatibility
- **Compressed Storage**: gzip compression for saving analysis results
- **Benchmarking Tools**: Performance testing with synthetic data generation
- **Optimized Algorithms**: Enhanced PCA, neighbor computation, and clustering methods

## Installation

### Prerequisites
```bash
# Install required Python packages
pip install scanpy pandas numpy matplotlib mcp fastmcp scipy scikit-learn numba

# Optional: for enhanced clustering methods
pip install leidenalg python-louvain
```

### Setup
1. Clone or download this repository
2. Save the MCP server code as `sc_analysis_mcp.py`
3. Run the server:
```bash
python sc_analysis_mcp.py
```

## Quick Start

### Basic Analysis Workflow

```python
# 1. Load your single-cell data
await load_data("path/to/your/data.h5ad", optimize_memory=True)

# 2. Preprocess the data
await preprocess_data(
    min_genes=200,
    min_cells=3,
    normalize=True,
    n_hvgs=2000,
    use_batch_correction=False
)

# 3. Run clustering
await run_clustering(method="leiden", resolution=0.5)

# 4. Find marker genes
await find_markers(n_genes=10)

# 5. Visualize results
await plot_umap(color_by="leiden")
```

### Trajectory Analysis Workflow

```python
# 1. Run Monocle3-style trajectory analysis
await run_monocle3_trajectory(
    root_cell_selection="auto",
    use_partition=True,
    resolution=0.5
)

# 2. Find genes varying along pseudotime
await find_trajectory_genes(
    n_genes=100,
    branch_specific=True
)

# 3. Create advanced trajectory visualizations
await plot_trajectory_advanced(
    color_by=['monocle3_pseudotime', 'monocle3_branch'],
    plot_type="combined",
    show_branch_points=True
)

# 4. Save your analysis
await save_data("trajectory_analysis.h5ad", compress=True)
```

## API Reference

### Core Functions

#### `load_data(file_path, optimize_memory=True)`
Load single-cell data from various formats with memory optimization options.

#### `preprocess_data(min_genes=200, min_cells=3, normalize=True, n_hvgs=2000, use_batch_correction=False, batch_key=None)`
Comprehensive preprocessing including QC filtering, normalization, and dimensionality reduction.

#### `run_clustering(method="louvain", resolution=0.5)`
Perform cell clustering using different algorithms with automatic fallbacks.

#### `find_markers(group_by=None, n_genes=10, method='wilcoxon')`
Identify marker genes for each cluster using statistical testing.

### Trajectory Analysis Functions

#### `run_monocle3_trajectory(root_cell_selection="auto", root_cell_idx=None, root_cluster=None, use_partition=True, resolution=0.5)`
Advanced trajectory inference with Monocle3-like capabilities.

#### `find_trajectory_genes(n_genes=100, branch_specific=True, test_branches=None, min_expr_frac=0.1)`
Identify genes that vary along the trajectory and between branches.

#### `plot_trajectory_advanced(color_by=None, plot_type="combined", save_path=None, show_branch_points=True, cell_size=1.0)`
Create sophisticated trajectory visualizations with multiple panel options.

### Utility Functions

#### `benchmark_performance(n_cells=1000, n_genes=5000)`
Test the performance of the analysis pipeline on synthetic data.

#### `save_data(file_path, compress=True)`
Save the complete analysis state with optional compression.

#### `get_data_summary()`
Get a comprehensive summary of loaded data and completed analyses.

## Performance Optimizations

### Recent Improvements (Latest Update)

1. **Monocle3-style Trajectory Analysis**
   - Partition-based cell grouping
   - Enhanced principal graph construction
   - Automatic branch detection and analysis
   - Root cell selection flexibility
   - Branch-specific gene identification

2. **Large Dataset Optimization**
   - Memory-efficient data structures
   - Sparse matrix operations
   - MiniBatch algorithms for clustering
   - Compressed data storage
   - Dynamic parameter adjustment based on dataset size

### Benchmarking Results

The server can efficiently process:
- **Small datasets**: <1,000 cells, <10,000 genes (< 30 seconds)
- **Medium datasets**: 1,000-10,000 cells, 10,000-30,000 genes (1-5 minutes)
- **Large datasets**: 10,000+ cells, 30,000+ genes (5-20 minutes)

Run `benchmark_performance()` to test on your system.

## Supported File Formats

- **h5ad**: Native AnnData format (recommended)
- **csv**: Comma-separated values with cells as rows
- **10x Genomics**: Matrix Market format directories

## Dependencies

### Core Dependencies
- scanpy >= 1.9.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0
- numba >= 0.54.0
- mcp >= 0.1.0
- fastmcp >= 0.1.0

### Optional Dependencies
- leidenalg: For Leiden clustering
- python-louvain: For Louvain clustering

## Troubleshooting

### Common Issues

1. **Memory Errors**: Enable `optimize_memory=True` when loading data
2. **Clustering Failures**: Install leidenalg or python-louvain packages
3. **Slow Performance**: Reduce `n_hvgs` parameter or use MiniBatch algorithms
4. **Trajectory Errors**: Ensure preprocessing is complete before running trajectory analysis

### Getting Help

For issues or feature requests, please open an issue on the project repository.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate error handling.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this MCP server in your research, please cite:

```
Single-Cell Analysis MCP Server. (2025). 
Available at: https://github.com/your-org/sc-analysis-mcp
```

## Acknowledgments

- Built on top of scanpy and the Python single-cell ecosystem
- Trajectory analysis inspired by Monocle3 methodology
- Model Context Protocol by Anthropic