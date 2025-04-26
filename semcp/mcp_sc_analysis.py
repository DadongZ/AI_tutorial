# mcp_sc_analysis.py (updated)

import scanpy as sc
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional

class MCPSingleCellAnalysis:
    def __init__(self):
        """Initialize the MCP for single cell analysis"""
        self.adata = None
        self.components = {}
        self.data_sources = []
        self.results = {}
    
    def register_component(self, name: str, component: Any):
        """Register a component with the MCP"""
        self.components[name] = component
    
    def load_data(self, adata: sc.AnnData):
        """Load single cell data for analysis"""
        self.adata = adata
        
        # Update components with data
        for component_name, component in self.components.items():
            if hasattr(component, 'load_data'):
                component.load_data(self.adata)
    
    def preprocess_data(self, **kwargs):
        """Preprocess data for analysis"""
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Basic scanpy preprocessing
        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        
        # Normalize data
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        # Scale data
        sc.pp.scale(self.adata, max_value=10)
        
        # Run PCA
        sc.tl.pca(self.adata, svd_solver='arpack')
        
        # Store preprocessing results
        self.results['preprocessing'] = {
            'n_cells': self.adata.n_obs,
            'n_genes': self.adata.n_vars,
            'hvg': self.adata.var['highly_variable'].sum()
        }
        
        return self.adata
    
    def run_trajectory_analysis(self, **kwargs):
        """Run trajectory analysis using Monocle3"""
        if 'monocle' not in self.components:
            raise ValueError("Monocle3 component not registered. Please register it first.")
        
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Run Monocle3 analysis
        monocle = self.components['monocle']
        cds = monocle.run_analysis(self.adata, **kwargs)
        
        # Store results
        self.results['trajectory'] = {
            'cds': cds,
            'umap_coords': monocle.get_umap_coordinates(cds),
            'pseudotime': monocle.get_pseudotime(cds)
        }
        
        # Add pseudotime to adata
        self.adata.obs['pseudotime'] = self.results['trajectory']['pseudotime']
        
        return self.results['trajectory']
    
    def discover_insights(self, condition_key='condition', **kwargs):
        """Run deep insight discovery"""
        if 'deep_insight' not in self.components:
            raise ValueError("Deep Insight component not registered. Please register it first.")
        
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Get deep insight component
        deep_insight = self.components['deep_insight']
        deep_insight.load_data(self.adata)
        
        # Preprocess data for deep learning
        preprocessed_data = deep_insight.preprocess_data()
        
        # Build and train model
        input_dim = preprocessed_data.X.shape[1]
        model = deep_insight.build_autoencoder(input_dim)
        
        # Train model
        X = preprocessed_data.X.toarray() if isinstance(preprocessed_data.X, np.ndarray) else preprocessed_data.X
        history = deep_insight.train_model(X, **kwargs)
        
        # Identify differential patterns
        patterns = deep_insight.identify_differential_patterns(condition_key)
        
        # Integrate with knowledge base
        if 'knowledge_base' in self.components:
            enrichment_results = deep_insight.integrate_knowledge()
        
        # Store results
        self.results['deep_insight'] = {
            'patterns': patterns,
            'model': model,
            'history': history
        }
        
        return self.results['deep_insight']
    
    def visualize_results(self, plot_type='umap', **kwargs):
        """Visualize analysis results"""
        if 'visualization' not in self.components:
            raise ValueError("Visualization component not registered. Please register it first.")
        
        visualization = self.components['visualization']
        
        if plot_type == 'umap':
            if self.adata is None:
                raise ValueError("No data loaded. Please load data first.")
            return visualization.plot_umap(self.adata, **kwargs)
        
        elif plot_type == 'trajectory':
            if 'trajectory' not in self.results:
                raise ValueError("Trajectory analysis not run. Please run trajectory analysis first.")
            return visualization.plot_trajectory(self.results['trajectory']['cds'], **kwargs)
        
        elif plot_type == 'patterns':
            if 'deep_insight' not in self.results:
                raise ValueError("Deep insight discovery not run. Please run discover_insights first.")
            return self.components['deep_insight'].visualize_patterns(**kwargs)
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def generate_summary(self):
        """Generate summary of analysis results"""
        summary = {
            "data": {
                "n_cells": self.adata.n_obs if self.adata is not None else 0,
                "n_genes": self.adata.n_vars if self.adata is not None else 0
            }
        }
        
        # Add preprocessing results if available
        if 'preprocessing' in self.results:
            summary["preprocessing"] = self.results['preprocessing']
        
        # Add trajectory results if available
        if 'trajectory' in self.results:
            summary["trajectory"] = {
                "num_branches": len(np.unique(self.results['trajectory']['cds'].clusters)),
                "pseudotime_range": [float(min(self.results['trajectory']['pseudotime'])), 
                                     float(max(self.results['trajectory']['pseudotime']))]
            }
        
        # Add deep insight results if available
        if 'deep_insight' in self.results and 'patterns' in self.results['deep_insight']:
            if 'deep_insight' in self.components:
                summary["insights"] = self.components['deep_insight'].summarize_insights()
        
        return summary
    
    def search_knowledge_base(self, query: str):
        """Search the knowledge base for information"""
        if 'knowledge_base' not in self.components:
            raise ValueError("Knowledge base component not registered. Please register it first.")
        
        knowledge_base = self.components['knowledge_base']
        results = knowledge_base.search(query)
        
        return results
    
    def process_query(self, query: str):
        """Process a natural language query from the user"""
        # Basic query handling - extend with LLM integration
        if "load" in query.lower() and "data" in query.lower():
            return "Please provide a file path to load data."
        
        elif "preprocess" in query.lower():
            try:
                self.preprocess_data()
                return "Data preprocessed successfully."
            except Exception as e:
                return f"Error preprocessing data: {str(e)}"
        
        elif "trajectory" in query.lower() or "pseudotime" in query.lower():
            try:
                results = self.run_trajectory_analysis()
                return "Trajectory analysis completed successfully."
            except Exception as e:
                return f"Error running trajectory analysis: {str(e)}"
        
        elif "insight" in query.lower() or "pattern" in query.lower():
            try:
                condition_key = "condition"  # Default - can be extracted from query
                results = self.discover_insights(condition_key)
                return "Deep insight analysis completed successfully."
            except Exception as e:
                return f"Error discovering insights: {str(e)}"
        
        elif "visualize" in query.lower() or "plot" in query.lower():
            try:
                if "umap" in query.lower():
                    self.visualize_results(plot_type='umap')
                elif "trajectory" in query.lower():
                    self.visualize_results(plot_type='trajectory')
                elif "pattern" in query.lower():
                    self.visualize_results(plot_type='patterns')
                else:
                    self.visualize_results()
                return "Visualization generated."
            except Exception as e:
                return f"Error generating visualization: {str(e)}"
        
        elif "summary" in query.lower():
            try:
                summary = self.generate_summary()
                return f"Analysis summary: {summary}"
            except Exception as e:
                return f"Error generating summary: {str(e)}"
        
        elif "search" in query.lower() or "knowledge" in query.lower():
            try:
                # Extract search terms
                search_terms = query.lower().split("search")[1].strip()
                results = self.search_knowledge_base(search_terms)
                return f"Knowledge base search results: {results}"
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        else:
            return "I don't understand that query. Try asking about loading data, preprocessing, trajectory analysis, discovering insights, visualization, or generating a summary."