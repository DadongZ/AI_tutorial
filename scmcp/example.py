
import scanpy as sc
import matplotlib.pyplot as plt
from mcp_sc_analysis import MCPSingleCellAnalysis
from monocle3 import Monocle3Analysis
from knowledge_integration import KnowledgeIntegration
from deep_insight import DeepInsightDiscovery
from visualization import Visualization

def run_example():
    """Run an example analysis"""
    print("Loading example data...")
    # Using scanpy's example dataset
    adata = sc.datasets.pbmc3k()
    
    # Initialize MCP
    print("Initializing MCP...")
    mcp = MCPSingleCellAnalysis()
    
    # Load data
    print("Loading data into MCP...")
    mcp.load_data(adata)
    
    # Initialize components
    print("Initializing components...")
    knowledge_base = KnowledgeIntegration()
    monocle = Monocle3Analysis()
    deep_insight = DeepInsightDiscovery(knowledge_base=knowledge_base)
    visualization = Visualization()
    
    # Register components with MCP
    mcp.register_component('knowledge_base', knowledge_base)
    mcp.register_component('monocle', monocle)
    mcp.register_component('deep_insight', deep_insight)
    mcp.register_component('visualization', visualization)
    
    # Preprocess data
    print("Preprocessing data...")
    mcp.preprocess_data()
    
    # Simulate a condition
    print("Simulating experimental conditions...")
    import numpy as np
    adata.obs['condition'] = np.random.choice(['control', 'treatment'], size=adata.n_obs, p=[0.5, 0.5])
    
    # Run trajectory analysis
    print("Running trajectory analysis...")
    try:
        trajectory_results = mcp.run_trajectory_analysis()
        print("Trajectory analysis complete.")
    except Exception as e:
        print(f"Error in trajectory analysis: {str(e)}")
    
    # Discover insights
    print("Discovering insights...")
    try:
        insight_results = mcp.discover_insights(condition_key='condition')
        print("Insight discovery complete.")
    except Exception as e:
        print(f"Error in insight discovery: {str(e)}")
    
    # Visualize results
    print("Generating visualizations...")
    try:
        fig = mcp.visualize_results(plot_type='umap', color='condition')
        plt.savefig('./output/example_umap.png')
        plt.close()
        print("Visualization saved to ./output/example_umap.png")
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
    
    # Generate summary
    print("Generating summary...")
    try:
        summary = mcp.generate_summary()
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Cells: {summary['data']['n_cells']}")
        print(f"Genes: {summary['data']['n_genes']}")
        if 'insights' in summary:
            print("Insights discovered.")
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
    
    print("\nExample complete. Check the ./output directory for results.")

if __name__ == "__main__":
    run_example()