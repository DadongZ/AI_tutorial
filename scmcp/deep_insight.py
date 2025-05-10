# deep_insight.py

import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt
import seaborn as sns
from knowledge_integration import KnowledgeIntegration

class DeepInsightDiscovery:
    def __init__(self, adata=None, knowledge_base=None):
        """
        Initialize the Deep Insight Discovery Engine
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data matrix with single cell data
        knowledge_base : KnowledgeIntegration
            Knowledge integration object with pathway, gene ontology, etc.
        """
        self.adata = adata
        self.knowledge_base = knowledge_base if knowledge_base else KnowledgeIntegration()
        self.patterns = {}
        self.model = None
    
    def load_data(self, adata):
        """Load single cell data for analysis"""
        self.adata = adata
    
    def preprocess_data(self):
        """Preprocess data for deep learning"""
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Filter out genes with low expression
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        # Normalize data
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        
        # Select highly variable genes
        sc.pp.highly_variable_genes(self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        self.adata = self.adata[:, self.adata.var.highly_variable]
        
        return self.adata
    
    def build_autoencoder(self, input_dim, encoding_dim=128, hidden_dims=[512, 256]):
        """
        Build an autoencoder for feature extraction and pattern detection
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of genes)
        encoding_dim : int
            Dimension of the latent space
        hidden_dims : list
            List of dimensions for hidden layers
        """
        # Encoder
        inputs = Input(shape=(input_dim,), name='input')
        x = inputs
        
        # Add hidden layers
        for dim in hidden_dims:
            x = Dense(dim, activation='relu')(x)
            x = Dropout(0.2)(x)
        
        # Bottleneck layer
        encoded = Dense(encoding_dim, activation='relu', name='encoded')(x)
        
        # Create encoder model
        self.encoder = Model(inputs, encoded, name='encoder')
        
        # Decoder
        x = encoded
        
        # Add hidden layers in reverse order
        for dim in reversed(hidden_dims):
            x = Dense(dim, activation='relu')(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        decoded = Dense(input_dim, activation='sigmoid', name='decoded')(x)
        
        # Create autoencoder model
        self.model = Model(inputs, decoded, name='autoencoder')
        
        # Compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model
    
    def train_model(self, X, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the deep learning model
        
        Parameters:
        -----------
        X : array
            Input data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        """
        if self.model is None:
            self.build_autoencoder(input_dim=X.shape[1])
        
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def extract_features(self, X):
        """Extract latent features using the trained encoder"""
        if self.encoder is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        latent_features = self.encoder.predict(X)
        return latent_features
    
    def identify_differential_patterns(self, condition_key='condition', control_group=None):
        """
        Identify patterns that differentiate between conditions
        
        Parameters:
        -----------
        condition_key : str
            Key in adata.obs that specifies the condition
        control_group : str
            Name of the control group in the condition
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Get condition groups
        conditions = self.adata.obs[condition_key].unique()
        if control_group is None:
            control_group = conditions[0]
        
        # Extract expression matrix
        X = self.adata.X.toarray() if isinstance(self.adata.X, np.ndarray) else self.adata.X
        
        # Get latent representation if model is trained
        if self.model is not None:
            latent_features = self.extract_features(X)
            
            # Add latent features to adata
            self.adata.obsm['latent_features'] = latent_features
            
            # Find top differential features in latent space
            sc.tl.rank_genes_groups(self.adata, condition_key, method='wilcoxon')
            
            # Store differential patterns
            self.patterns['differential_genes'] = pd.DataFrame(
                self.adata.uns['rank_genes_groups']['names']
            )
            
            # Use latent features for further analysis
            pca = PCA(n_components=10)
            pca_result = pca.fit_transform(latent_features)
            self.adata.obsm['X_pca'] = pca_result
            
            # Compute UMAP for visualization
            sc.pp.neighbors(self.adata, use_rep='X_pca')
            sc.tl.umap(self.adata)
        else:
            # Perform standard differential expression
            sc.tl.rank_genes_groups(self.adata, condition_key, method='wilcoxon')
            
            # Store differential patterns
            self.patterns['differential_genes'] = pd.DataFrame(
                self.adata.uns['rank_genes_groups']['names']
            )
        
        return self.patterns
    
    def integrate_knowledge(self, pattern_key='differential_genes', top_n=100):
        """
        Integrate identified patterns with knowledge base
        
        Parameters:
        -----------
        pattern_key : str
            Key to access the patterns dictionary
        top_n : int
            Number of top genes to use for knowledge integration
        """
        if pattern_key not in self.patterns:
            raise ValueError(f"Pattern key '{pattern_key}' not found")
        
        # Extract top differentially expressed genes
        diff_genes = self.patterns[pattern_key]
        
        # Get top genes for each condition
        gene_lists = {}
        for col in diff_genes.columns:
            gene_lists[col] = diff_genes[col].head(top_n).tolist()
        
        # Perform enrichment analysis using knowledge base
        enrichment_results = {}
        for condition, genes in gene_lists.items():
            enrichment_results[condition] = {
                'pathway_enrichment': self.knowledge_base.pathway_analysis(genes),
                'go_enrichment': self.knowledge_base.go_enrichment(genes),
                'disease_association': self.knowledge_base.disease_association(genes)
            }
        
        self.patterns['enrichment_results'] = enrichment_results
        return enrichment_results
    
    def visualize_patterns(self, pattern_type='umap', condition_key='condition'):
        """
        Visualize identified patterns
        
        Parameters:
        -----------
        pattern_type : str
            Type of visualization (umap, heatmap, enrichment)
        condition_key : str
            Key in adata.obs that specifies the condition
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")
        
        plt.figure(figsize=(10, 8))
        
        if pattern_type == 'umap':
            # Visualize UMAP with conditions
            sc.pl.umap(self.adata, color=condition_key, show=False)
            plt.title('UMAP visualization colored by condition')
            
        elif pattern_type == 'heatmap':
            # Get top differential genes
            if 'differential_genes' not in self.patterns:
                raise ValueError("Differential genes not computed. Run identify_differential_patterns first.")
            
            diff_genes = self.patterns['differential_genes']
            top_genes = []
            for col in diff_genes.columns:
                top_genes.extend(diff_genes[col].head(10).tolist())
            
            # Remove duplicates
            top_genes = list(set(top_genes))
            
            # Create heatmap
            sc.pl.heatmap(self.adata, top_genes, groupby=condition_key, show=False)
            plt.title('Heatmap of top differential genes across conditions')
            
        elif pattern_type == 'enrichment':
            # Visualize enrichment results
            if 'enrichment_results' not in self.patterns:
                raise ValueError("Enrichment results not computed. Run integrate_knowledge first.")
            
            enrichment = self.patterns['enrichment_results']
            condition = list(enrichment.keys())[0]
            
            # Plot top pathways
            pathways = enrichment[condition]['pathway_enrichment']
            if pathways is not None and len(pathways) > 0:
                if isinstance(pathways, pd.DataFrame):
                    top_pathways = pathways.head(10)
                    sns.barplot(x='p_value', y='term', data=top_pathways)
                    plt.title(f'Top enriched pathways for {condition}')
                    plt.tight_layout()
            
        plt.tight_layout()
        return plt.gcf()
    
    def summarize_insights(self):
        """Generate a summary of discoveries from the analysis"""
        if not self.patterns:
            raise ValueError("No patterns identified. Run identify_differential_patterns first.")
        
        summary = {
            "differential_genes": {},
            "pathways": {},
            "diseases": {}
        }
        
        # Summarize differential genes
        if 'differential_genes' in self.patterns:
            diff_genes = self.patterns['differential_genes']
            for col in diff_genes.columns:
                summary["differential_genes"][col] = diff_genes[col].head(10).tolist()
        
        # Summarize enrichment results
        if 'enrichment_results' in self.patterns:
            enrichment = self.patterns['enrichment_results']
            for condition, results in enrichment.items():
                # Summarize pathways
                if 'pathway_enrichment' in results and results['pathway_enrichment'] is not None:
                    pathways = results['pathway_enrichment']
                    if isinstance(pathways, pd.DataFrame) and not pathways.empty:
                        summary["pathways"][condition] = pathways.head(5)['term'].tolist()
                
                # Summarize disease associations
                if 'disease_association' in results and results['disease_association'] is not None:
                    diseases = results['disease_association']
                    if isinstance(diseases, pd.DataFrame) and not diseases.empty:
                        summary["diseases"][condition] = diseases.head(5)['term'].tolist()
        
        return summary