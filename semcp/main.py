# main.py

import os
import argparse
import scanpy as sc
from interface import UserInterface
from monocle3 import Monocle3Analysis
from knowledge_integration import KnowledgeIntegration
from deep_insight import DeepInsightDiscovery
from visualization import Visualization
from mcp_sc_analysis import MCPSingleCellAnalysis

def main():
    """Main function to run the MCP for single cell analysis"""
    parser = argparse.ArgumentParser(description='MCP for Single Cell Analysis')
    parser.add_argument('--data', type=str, help='Path to single cell data file')
    parser.add_argument('--condition', type=str, default='condition', help='Column name for condition')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--knowledge', type=str, default=None, help='Path to knowledge base files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize MCP
    mcp = MCPSingleCellAnalysis()
    
    # Initialize UI
    ui = UserInterface(mcp)
    
    # Load data
    if args.data:
        adata = sc.read(args.data)
        mcp.load_data(adata)
        print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes.")
    else:
        print("No data file provided. Please use the UI to load data.")
    
    # Initialize components
    knowledge_base = KnowledgeIntegration()
    if args.knowledge:
        knowledge_base.load_database(args.knowledge)
    
    monocle = Monocle3Analysis()
    deep_insight = DeepInsightDiscovery(knowledge_base=knowledge_base)
    visualization = Visualization()
    
    # Register components with MCP
    mcp.register_component('knowledge_base', knowledge_base)
    mcp.register_component('monocle', monocle)
    mcp.register_component('deep_insight', deep_insight)
    mcp.register_component('visualization', visualization)
    
    # Start UI
    ui.start()

if __name__ == "__main__":
    main()