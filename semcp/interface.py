# interface.py (updated)

import sys
import cmd
import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

class UserInterface(cmd.Cmd):
    intro = "Welcome to the Single Cell MCP. Type help or ? to list commands."
    prompt = "(scMCP) "
    
    def __init__(self, mcp):
        """Initialize the user interface with an MCP instance"""
        super().__init__()
        self.mcp = mcp
        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def do_load(self, arg):
        """Load single cell data file: load path/to/file.h5ad"""
        try:
            if not arg:
                print("Please provide a file path")
                return
            
            file_path = arg.strip()
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return
            
            adata = sc.read(file_path)
            self.mcp.load_data(adata)
            print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes.")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
    
    def do_preprocess(self, arg):
        """Preprocess loaded data"""
        try:
            adata = self.mcp.preprocess_data()
            print(f"Preprocessing complete. {adata.n_obs} cells, {adata.n_vars} genes.")
            print(f"Identified {adata.var['highly_variable'].sum()} highly variable genes.")
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
    
    def do_trajectory(self, arg):
        """Run trajectory analysis using Monocle3"""
        try:
            # Parse arguments
            args = {}
            if arg:
                arg_pairs = arg.strip().split()
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=')
                        args[key] = value
            
            results = self.mcp.run_trajectory_analysis(**args)
            print("Trajectory analysis complete.")
            print(f"Number of branches: {len(set(results['cds'].clusters))}")
            print(f"Pseudotime range: {min(results['pseudotime'])} - {max(results['pseudotime'])}")
        except Exception as e:
            print(f"Error running trajectory analysis: {str(e)}")
    
    def do_insights(self, arg):
        """Discover deep insights in the data"""
        try:
            # Parse arguments
            args = {}
            condition_key = 'condition'
            
            if arg:
                arg_pairs = arg.strip().split()
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=')
                        if key == 'condition':
                            condition_key = value
                        else:
                            args[key] = value
            
            results = self.mcp.discover_insights(condition_key, **args)
            print("Deep insight discovery complete.")
            
            if 'patterns' in results and 'differential_genes' in results['patterns']:
                diff_genes = results['patterns']['differential_genes']
                for col in diff_genes.columns:
                    print(f"Top 5 genes for {col}: {', '.join(diff_genes[col].head(5).tolist())}")
            
            if 'history' in results:
                print(f"Model training completed with final loss: {results['history'].history['loss'][-1]:.4f}")
        except Exception as e:
            print(f"Error discovering insights: {str(e)}")
    
    def do_visualize(self, arg):
        """Visualize results: visualize [type] [options]"""
        try:
            # Parse arguments
            plot_type = 'umap'  # default
            args = {}
            
            if arg:
                arg_parts = arg.strip().split()
                if arg_parts:
                    plot_type = arg_parts[0].lower()
                
                for part in arg_parts[1:]:
                    if '=' in part:
                        key, value = part.split('=')
                        args[key] = value
            
            fig = self.mcp.visualize_results(plot_type=plot_type, **args)
            
            # Save figure
            filename = f"{self.output_dir}/{plot_type}_visualization.png"
            plt.savefig(filename)
            plt.close()
            
            print(f"Visualization saved to {filename}")
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")
    
    def do_summary(self, arg):
        """Generate a summary of the analysis results"""
        try:
            summary = self.mcp.generate_summary()
            
            # Print summary
            print("\n=== ANALYSIS SUMMARY ===")
            
            # Data summary
            print(f"\nDATA:")
            print(f"  Cells: {summary['data']['n_cells']}")
            print(f"  Genes: {summary['data']['n_genes']}")
            
            # Preprocessing summary
            if 'preprocessing' in summary:
                print(f"\nPREPROCESSING:")
                print(f"  Highly variable genes: {summary['preprocessing']['hvg']}")
            
            # Trajectory summary
            if 'trajectory' in summary:
                print(f"\nTRAJECTORY:")
                print(f"  Number of branches: {summary['trajectory']['num_branches']}")
                print(f"  Pseudotime range: {summary['trajectory']['pseudotime_range'][0]:.2f} - {summary['trajectory']['pseudotime_range'][1]:.2f}")
            
            # Insights summary
            if 'insights' in summary:
                print(f"\nINSIGHTS:")
                
                # Print differential genes
                if 'differential_genes' in summary['insights']:
                    print(f"  Differential Genes:")
                    for condition, genes in summary['insights']['differential_genes'].items():
                        print(f"    {condition}: {', '.join(genes[:5])}")
                
                # Print pathways
                if 'pathways' in summary['insights']:
                    print(f"  Enriched Pathways:")
                    for condition, pathways in summary['insights']['pathways'].items():
                        print(f"    {condition}: {', '.join(pathways[:3])}")
                
                # Print diseases
                if 'diseases' in summary['insights']:
                    print(f"  Disease Associations:")
                    for condition, diseases in summary['insights']['diseases'].items():
                        print(f"    {condition}: {', '.join(diseases[:3])}")
            
            # Save summary to file
            summary_file = f"{self.output_dir}/analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(str(summary))
            
            print(f"\nSummary saved to {summary_file}")
        except Exception:
            print(f"\nSummary saved to {summary_file}")
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
    
    def do_search(self, arg):
        """Search knowledge base: search [query]"""
        try:
            if not arg:
                print("Please provide a search query")
                return
            
            results = self.mcp.search_knowledge_base(arg)
            print("\n=== KNOWLEDGE BASE SEARCH RESULTS ===")
            
            if not results:
                print("No results found.")
                return
            
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result.get('term', 'Unknown')}")
                if 'description' in result:
                    print(f"   {result['description']}")
                if 'source' in result:
                    print(f"   Source: {result['source']}")
        except Exception as e:
            print(f"Error searching knowledge base: {str(e)}")
    
    def do_query(self, arg):
        """Process a natural language query: query [your question here]"""
        if not arg:
            print("Please provide a query")
            return
        
        response = self.mcp.process_query(arg)
        print(response)
    
    def do_save(self, arg):
        """Save current analysis state: save [filename]"""
        try:
            filename = arg.strip() if arg else "mcp_analysis_state.h5ad"
            if not filename.endswith('.h5ad'):
                filename += '.h5ad'
            
            if self.mcp.adata is None:
                print("No data to save")
                return
            
            # Save AnnData object
            save_path = os.path.join(self.output_dir, filename)
            self.mcp.adata.write(save_path)
            
            print(f"Analysis state saved to {save_path}")
        except Exception as e:
            print(f"Error saving analysis state: {str(e)}")
    
    def do_exit(self, arg):
        """Exit the program"""
        print("Exiting MCP. Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the program"""
        return self.do_exit(arg)
    
    def start(self):
        """Start the UI"""
        self.cmdloop()