# Visualization Module for MCP
# Provides various visualization functions for single cell data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for generating visualizations from single cell data
    """
    
    def __init__(self, output_dir: str = "./figures", dpi: int = 300):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save figures
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set default style
        sns.set_style("whitegrid")
        
        logger.info(f"Visualization module initialized with output directory: {output_dir}")
    
    def get_palette(self, n_colors: int) -> List[str]:
        """
        Get a color palette for visualization
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of color hex codes
        """
        # Use different palettes based on number of colors
        if n_colors <= 10:
            return sns.color_palette("tab10", n_colors).as_hex()
        elif n_colors <= 20:
            return sns.color_palette("tab20", n_colors).as_hex()
        else:
            return sns.color_palette("husl", n_colors).as_hex()
    
    def plot_umap(self, adata, color_by: str, figsize: Tuple[int, int] = (10, 8), 
                 title: str = None, save_as: str = None, **kwargs) -> plt.Figure:
        """
        Plot UMAP visualization of single cell data
        
        Args:
            adata: AnnData object
            color_by: Feature to color by (cluster, gene, etc.)
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            **kwargs: Additional arguments to pass to sc.pl.umap
            
        Returns:
            Matplotlib figure
        """
        if 'X_umap' not in adata.obsm:
            raise ValueError("UMAP not computed. Run dim_reduction first.")
            
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set title if not provided
            if title is None:
                title = f"UMAP visualization colored by {color_by}"
                
            # Check if color_by is a gene or metadata
            if color_by in adata.obs.columns:
                # Color by categorical variable
                if adata.obs[color_by].dtype.name == 'category' or len(adata.obs[color_by].unique()) < 20:
                    palette = self.get_palette(len(adata.obs[color_by].unique()))
                    sc.pl.umap(adata, color=color_by, ax=ax, title=title, palette=palette, show=False, **kwargs)
                else:
                    # Continuous variable
                    sc.pl.umap(adata, color=color_by, ax=ax, title=title, color_map='viridis', show=False, **kwargs)
            elif color_by in adata.var_names:
                # Color by gene expression
                sc.pl.umap(adata, color=color_by, ax=ax, title=title, color_map='viridis', show=False, **kwargs)
            else:
                raise ValueError(f"Feature {color_by} not found in data")
                
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"UMAP visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in UMAP visualization: {str(e)}")
            raise
    
    def plot_trajectory(self, adata, color_by: str = 'dpt_pseudotime', 
                       figsize: Tuple[int, int] = (10, 8), title: str = None, 
                       save_as: str = None, **kwargs) -> plt.Figure:
        """
        Plot trajectory visualization of single cell data
        
        Args:
            adata: AnnData object
            color_by: Feature to color by (pseudotime, cluster, gene, etc.)
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            **kwargs: Additional arguments to pass to sc.pl.umap
            
        Returns:
            Matplotlib figure
        """
        if 'X_umap' not in adata.obsm:
            raise ValueError("UMAP not computed. Run dim_reduction first.")
            
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set title if not provided
            if title is None:
                title = f"Trajectory visualization colored by {color_by}"
                
            # Check if color_by is a gene or metadata
            if color_by in adata.obs.columns:
                # For pseudotime or continuous variables
                if color_by.endswith('_pseudotime') or adata.obs[color_by].dtype.name in ['float64', 'float32']:
                    sc.pl.umap(adata, color=color_by, ax=ax, title=title, 
                             color_map='viridis', show=False, **kwargs)
                else:
                    # Categorical variable
                    palette = self.get_palette(len(adata.obs[color_by].unique()))
                    sc.pl.umap(adata, color=color_by, ax=ax, title=title, 
                             palette=palette, show=False, **kwargs)
            elif color_by in adata.var_names:
                # Color by gene expression
                sc.pl.umap(adata, color=color_by, ax=ax, title=title, 
                         color_map='viridis', show=False, **kwargs)
            else:
                raise ValueError(f"Feature {color_by} not found in data")
                
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Trajectory visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in trajectory visualization: {str(e)}")
            raise
    
    def plot_heatmap(self, adata, var_names: List[str], groupby: str = 'leiden', 
                    figsize: Tuple[int, int] = (12, 10), title: str = None, 
                    save_as: str = None, **kwargs) -> plt.Figure:
        """
        Plot heatmap of gene expression
        
        Args:
            adata: AnnData object
            var_names: List of genes to include in heatmap
            groupby: Variable to group cells by
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            **kwargs: Additional arguments to pass to sc.pl.heatmap
            
        Returns:
            Matplotlib figure
        """
        if groupby not in adata.obs.columns:
            raise ValueError(f"Groupby variable {groupby} not found in data")
            
        try:
            # Create a copy to avoid modifying the original
            adata_subset = adata.copy()
            
            # Filter to only include specified genes
            var_names = [gene for gene in var_names if gene in adata.var_names]
            if len(var_names) == 0:
                raise ValueError("None of the specified genes found in data")
                
            # Set title if not provided
            if title is None:
                title = f"Expression heatmap of {len(var_names)} genes across {groupby}"
                
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Plot heatmap
            sc.pl.heatmap(adata_subset, var_names=var_names, groupby=groupby, 
                        show_gene_labels=True, standard_scale='var', 
                        title=title, show=False, **kwargs)
                
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Heatmap visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in heatmap visualization: {str(e)}")
            raise
            
    def plot_marker_violin(self, adata, genes: List[str], groupby: str = 'leiden',
                        figsize: Tuple[int, int] = (12, 10), title: str = None,
                        save_as: str = None, **kwargs) -> plt.Figure:
        """
        Plot violin plots of marker gene expression across groups
        
        Args:
            adata: AnnData object
            genes: List of genes to plot
            groupby: Variable to group cells by
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            **kwargs: Additional arguments to pass to sc.pl.violin
            
        Returns:
            Matplotlib figure
        """
        if groupby not in adata.obs.columns:
            raise ValueError(f"Groupby variable {groupby} not found in data")
            
        try:
            # Filter to only include specified genes that are in the dataset
            genes = [gene for gene in genes if gene in adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes found in data")
                
            # Set title if not provided
            if title is None:
                title = f"Expression of marker genes across {groupby}"
                
            # Create figure
            fig, axs = plt.subplots(len(genes), 1, figsize=figsize, constrained_layout=True)
            if len(genes) == 1:
                axs = [axs]
                
            # Plot each gene
            for i, gene in enumerate(genes):
                sc.pl.violin(adata, keys=gene, groupby=groupby, ax=axs[i], 
                           title=f"{gene} expression", show=False, **kwargs)
                
            # Set overall title
            fig.suptitle(title, fontsize=16)
                
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Violin plot visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in violin plot visualization: {str(e)}")
            raise
            
    def plot_dotplot(self, adata, var_names: List[str], groupby: str = 'leiden',
                   figsize: Tuple[int, int] = (12, 8), title: str = None,
                   save_as: str = None, **kwargs) -> plt.Figure:
        """
        Create a dot plot for visualizing gene expression across groups
        
        Args:
            adata: AnnData object
            var_names: List of genes to include in the plot
            groupby: Variable to group cells by
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            **kwargs: Additional arguments to pass to sc.pl.dotplot
            
        Returns:
            Matplotlib figure
        """
        if groupby not in adata.obs.columns:
            raise ValueError(f"Groupby variable {groupby} not found in data")
            
        try:
            # Filter to only include specified genes that are in the dataset
            var_names = [gene for gene in var_names if gene in adata.var_names]
            if len(var_names) == 0:
                raise ValueError("None of the specified genes found in data")
                
            # Set title if not provided
            if title is None:
                title = f"Expression dot plot across {groupby}"
                
            # Create figure and plot
            fig = plt.figure(figsize=figsize)
            sc.pl.dotplot(adata, var_names=var_names, groupby=groupby, 
                        title=title, show=False, **kwargs)
                
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Dot plot visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in dot plot visualization: {str(e)}")
            raise
    
    def plot_trajectory_heatmap(self, adata, genes: List[str], pseudotime_key: str = 'dpt_pseudotime',
                              n_bins: int = 20, figsize: Tuple[int, int] = (12, 10), 
                              title: str = None, save_as: str = None) -> plt.Figure:
        """
        Plot heatmap of gene expression along pseudotime trajectory
        
        Args:
            adata: AnnData object
            genes: List of genes to include in heatmap
            pseudotime_key: Key for pseudotime values in adata.obs
            n_bins: Number of bins for pseudotime
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        if pseudotime_key not in adata.obs.columns:
            raise ValueError(f"Pseudotime key {pseudotime_key} not found in data")
            
        try:
            # Filter to only include specified genes that are in the dataset
            genes = [gene for gene in genes if gene in adata.var_names]
            if len(genes) == 0:
                raise ValueError("None of the specified genes found in data")
                
            # Get pseudotime values and create bins
            pseudotime = adata.obs[pseudotime_key].values
            bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
            bin_assignments = np.digitize(pseudotime, bins) - 1
            
            # Create binned expression matrix
            bin_means = np.zeros((n_bins, len(genes)))
            for i in range(n_bins):
                mask = bin_assignments == i
                if np.sum(mask) > 0:
                    for j, gene in enumerate(genes):
                        bin_means[i, j] = np.mean(adata[:, gene].X.toarray().flatten()[mask])
            
            # Z-score normalize
            bin_means_z = (bin_means - np.mean(bin_means, axis=0)) / np.std(bin_means, axis=0)
            
            # Set title if not provided
            if title is None:
                title = f"Gene expression dynamics along pseudotime trajectory"
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot heatmap
            im = ax.imshow(bin_means_z.T, aspect='auto', cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Z-score of expression')
            
            # Set labels
            ax.set_xlabel('Pseudotime')
            ax.set_ylabel('Genes')
            
            # Set y-ticks to gene names
            ax.set_yticks(np.arange(len(genes)))
            ax.set_yticklabels(genes)
            
            # Set x-ticks to pseudotime bins
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.set_xticks(np.arange(0, n_bins, n_bins // 5))
            ax.set_xticklabels([f"{x:.2f}" for x in bin_centers[::n_bins//5]])
            
            # Set title
            ax.set_title(title)
            
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Trajectory heatmap saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in trajectory heatmap visualization: {str(e)}")
            raise
    
    def plot_pathway_enrichment(self, pathway_results: Dict, 
                              figsize: Tuple[int, int] = (10, 8),
                              title: str = None, save_as: str = None,
                              max_terms: int = 15) -> plt.Figure:
        """
        Plot pathway enrichment results
        
        Args:
            pathway_results: Dictionary with pathway enrichment results
            figsize: Figure size
            title: Plot title
            save_as: Filename to save figure
            max_terms: Maximum number of pathways to display
            
        Returns:
            Matplotlib figure
        """
        try:
            # Extract pathway data
            if 'results' not in pathway_results:
                raise ValueError("Invalid pathway results format")
                
            pathways = pathway_results['results']
            if not pathways:
                raise ValueError("No pathways found in results")
                
            # Convert to DataFrame
            pathway_df = pd.DataFrame(pathways)
            
            # Sort by p-value
            pathway_df = pathway_df.sort_values('p_value')
            
            # Limit to max_terms
            if len(pathway_df) > max_terms:
                pathway_df = pathway_df.head(max_terms)
                
            # Set title if not provided
            if title is None:
                db = pathway_results.get('database', 'pathway')
                title = f"Top enriched {db} pathways"
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot horizontal bar chart
            bars = ax.barh(pathway_df['name'], -np.log10(pathway_df['p_value']), color='skyblue')
            
            # Add p-value annotations
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + 0.3, 
                      bar.get_y() + bar.get_height()/2, 
                      f"p={pathway_df.iloc[i]['p_value']:.1e}",
                      va='center')
                
            # Set labels
            ax.set_xlabel('-log10(p-value)')
            ax.set_ylabel('Pathway')
            
            # Adjust y-axis to show pathway names
            plt.tight_layout()
            
            # Set title
            ax.set_title(title)
            
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Pathway enrichment visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in pathway enrichment visualization: {str(e)}")
            raise
    
    def plot_cluster_summary(self, adata, cluster_knowledge: Dict, 
                           cluster_key: str = 'leiden',
                           figsize: Tuple[int, int] = (15, 10),
                           save_as: str = None) -> plt.Figure:
        """
        Create a summary visualization of cluster characteristics
        
        Args:
            adata: AnnData object
            cluster_knowledge: Dictionary with knowledge integration results per cluster
            cluster_key: Key for cluster information in adata.obs
            figsize: Figure size
            save_as: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            # Get unique clusters
            clusters = sorted(list(cluster_knowledge.keys()))
            n_clusters = len(clusters)
            
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Create grid of subplots
            gs = fig.add_gridspec(2, 2)
            
            # 1. UMAP with clusters
            ax1 = fig.add_subplot(gs[0, 0])
            sc.pl.umap(adata, color=cluster_key, ax=ax1, title="Cell Clusters", show=False)
            
            # 2. Cluster cell counts
            ax2 = fig.add_subplot(gs[0, 1])
            cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
            ax2.bar(cluster_counts.index, cluster_counts.values, color=self.get_palette(n_clusters))
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Number of cells')
            ax2.set_title('Cell count per cluster')
            
            # 3. Pathway/GO term summary
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Create a table of top GO terms or pathways per cluster
            cell_text = []
            for cluster in clusters:
                if 'top_go_terms' in cluster_knowledge[cluster] and cluster_knowledge[cluster]['top_go_terms']:
                    term = cluster_knowledge[cluster]['top_go_terms'][0]
                    cell_text.append([cluster, term['name'], f"{term['p_value']:.2e}"])
                elif 'top_pathways' in cluster_knowledge[cluster] and cluster_knowledge[cluster]['top_pathways']:
                    term = cluster_knowledge[cluster]['top_pathways'][0]
                    cell_text.append([cluster, term['name'], f"{term['p_value']:.2e}"])
                else:
                    cell_text.append([cluster, "N/A", "N/A"])
            
            ax3.axis('tight')
            ax3.axis('off')
            table = ax3.table(cellText=cell_text, 
                            colLabels=['Cluster', 'Top GO term/Pathway', 'p-value'],
                            loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax3.set_title('Top biological feature per cluster')
            
            # 4. Disease association summary
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Create a table of top diseases per cluster
            cell_text = []
            for cluster in clusters:
                if 'top_diseases' in cluster_knowledge[cluster] and cluster_knowledge[cluster]['top_diseases']:
                    disease = cluster_knowledge[cluster]['top_diseases'][0]
                    cell_text.append([cluster, disease['name'], f"{disease['score']:.2f}"])
                else:
                    cell_text.append([cluster, "N/A", "N/A"])
                    
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=cell_text, 
                            colLabels=['Cluster', 'Top disease association', 'Score'],
                            loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax4.set_title('Top disease association per cluster')
            
            # Add overall title
            fig.suptitle('Cluster Summary Dashboard', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure if path provided
            if save_as:
                save_path = self.output_dir / save_as
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Cluster summary visualization saved to {save_path}")
                
            return fig
            
        except Exception as e:
            logger.error(f"Error in cluster summary visualization: {str(e)}")
            raise
    
    def create_visualization_report(self, adata, knowledge_results: Dict, 
                                  output_file: str = "visualization_report.pdf"):
        """
        Create a comprehensive visualization report with multiple plots
        
        Args:
            adata: AnnData object
            knowledge_results: Dictionary with knowledge integration results
            output_file: Filename for the PDF report
            
        Returns:
            Path to the saved report
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        try:
            # Prepare output path
            output_path = self.output_dir / output_file
            
            # Create PDF
            with PdfPages(output_path) as pdf:
                # 1. UMAP visualization
                if 'leiden' in adata.obs.columns:
                    fig = self.plot_umap(adata, color_by='leiden', title="Cell Clusters")
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # 2. Trajectory visualization if available
                if 'dpt_pseudotime' in adata.obs.columns:
                    fig = self.plot_trajectory(adata, color_by='dpt_pseudotime', 
                                            title="Pseudotime Trajectory")
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # 3. Marker gene heatmap if available
                if 'rank_genes_groups' in adata.uns:
                    # Get top marker genes for each cluster
                    top_markers = []
                    for group in adata.uns['rank_genes_groups']['names'].dtype.names:
                        markers = [gene for gene in adata.uns['rank_genes_groups']['names'][group][:5]]
                        top_markers.extend(markers)
                    
                    if top_markers:
                        fig = self.plot_heatmap(adata, var_names=top_markers, groupby='leiden',
                                             title="Top Marker Genes per Cluster")
                        pdf.savefig(fig)
                        plt.close(fig)
                
                # 4. Pathway enrichment plots if available
                if 'pathway_enrichment' in knowledge_results:
                    for cluster, knowledge in knowledge_results.items():
                        if 'pathway_enrichment' in knowledge:
                            fig = self.plot_pathway_enrichment(
                                knowledge['pathway_enrichment'],
                                title=f"Pathway Enrichment for Cluster {cluster}"
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                
                # 5. Add cluster summary dashboard if available
                if 'leiden' in adata.obs.columns and knowledge_results:
                    # Summarize the knowledge results per cluster
                    from collections import defaultdict
                    summary = defaultdict(dict)
                    
                    for cluster, knowledge in knowledge_results.items():
                        # Extract top GO terms
                        if 'gene_ontology' in knowledge and 'results' in knowledge['gene_ontology']:
                            go_terms = knowledge['gene_ontology']['results']
                            summary[cluster]['top_go_terms'] = [
                                {"name": term["name"], "p_value": term["p_value"]} 
                                for term in sorted(go_terms, key=lambda x: x["p_value"])[:3]
                            ]
                        
                        # Extract top diseases
                        if 'disease_association' in knowledge and 'results' in knowledge['disease_association']:
                            diseases = knowledge['disease_association']['results']
                            summary[cluster]['top_diseases'] = [
                                {"name": disease["disease"], "score": disease["score"]} 
                                for disease in sorted(diseases, key=lambda x: x["score"], reverse=True)[:3]
                            ]
                    
                    # Create summary visualization
                    fig = self.plot_cluster_summary(adata, summary)
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Add metadata to PDF
                d = pdf.infodict()
                d['Title'] = 'Single Cell RNA-Seq Analysis Report'
                d['Author'] = 'MCP Single Cell'
                
            logger.info(f"Visualization report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization report: {str(e)}")
            raise