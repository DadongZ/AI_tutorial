# Knowledge Integration Module for MCP
# Integrates external databases and knowledge resources

import os
import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeIntegration:
    """
    Class for integrating external knowledge sources with single cell analysis
    """
    
    def __init__(self, api_keys: Dict[str, str] = None, cache_dir: str = None):
        """
        Initialize Knowledge Integration module
        
        Args:
            api_keys: Dictionary of API keys for different services
            cache_dir: Directory to cache results from external APIs
        """
        self.api_keys = api_keys or {}
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".mcp_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize available knowledge sources
        self.knowledge_sources = {
            "gene_ontology": self.query_gene_ontology,
            "pathway_enrichment": self.query_pathway_enrichment,
            "disease_association": self.query_disease_association,
            "drug_target": self.query_drug_targets,
            "literature": self.query_literature
        }
        
        logger.info("Knowledge Integration module initialized")
    
    def query_gene_ontology(self, gene_list: List[str], organism: str = "human") -> Dict:
        """
        Query Gene Ontology for enrichment of gene list
        
        Args:
            gene_list: List of gene symbols to query
            organism: Organism to query (human, mouse, etc.)
            
        Returns:
            Dictionary with enrichment results
        """
        logger.info(f"Querying Gene Ontology for {len(gene_list)} genes")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"go_{organism}_{hash(tuple(sorted(gene_list)))}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # In production, use a real API call
        # For this example, we'll simulate results
        go_terms = [
            {"id": "GO:0006955", "name": "immune response", "p_value": 0.001, "genes": gene_list[:5]},
            {"id": "GO:0007165", "name": "signal transduction", "p_value": 0.003, "genes": gene_list[2:7]},
            {"id": "GO:0006915", "name": "apoptotic process", "p_value": 0.02, "genes": gene_list[1:4]},
            {"id": "GO:0008283", "name": "cell proliferation", "p_value": 0.05, "genes": gene_list[3:8]},
            {"id": "GO:0000082", "name": "G1/S transition of mitotic cell cycle", "p_value": 0.07, "genes": gene_list[2:5]},
            {"id": "GO:0007049", "name": "cell cycle", "p_value": 0.08, "genes": gene_list[4:9]}
        ]
        
        # Example of what a real API call might look like:
        # try:
        #     response = requests.post(
        #         "https://api.geneontology.org/api/enrichment",
        #         headers={"Authorization": f"Bearer {self.api_keys.get('go', '')}"},
        #         json={"genes": gene_list, "organism": organism}
        #     )
        #     if response.status_code == 200:
        #         go_terms = response.json()["results"]
        #     else:
        #         logger.error(f"Error querying Gene Ontology API: {response.text}")
        #         go_terms = []
        # except Exception as e:
        #     logger.error(f"Error in Gene Ontology query: {str(e)}")
        #     go_terms = []
        
        # Cache results
        results = {
            "query_genes": gene_list,
            "organism": organism,
            "results": go_terms
        }
        
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
    
    def query_pathway_enrichment(self, gene_list: List[str], database: str = "kegg") -> Dict:
        """
        Query pathway databases for enrichment analysis
        
        Args:
            gene_list: List of gene symbols to query
            database: Pathway database to query (kegg, reactome, etc.)
            
        Returns:
            Dictionary with pathway enrichment results
        """
        logger.info(f"Querying {database} pathways for {len(gene_list)} genes")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"pathway_{database}_{hash(tuple(sorted(gene_list)))}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Simulate results for different databases
        if database.lower() == "kegg":
            pathways = [
                {"id": "hsa04060", "name": "Cytokine-cytokine receptor interaction", "p_value": 0.001, "genes": gene_list[:5]},
                {"id": "hsa04151", "name": "PI3K-Akt signaling pathway", "p_value": 0.008, "genes": gene_list[2:7]},
                {"id": "hsa04010", "name": "MAPK signaling pathway", "p_value": 0.015, "genes": gene_list[1:4]}
            ]
        elif database.lower() == "reactome":
            pathways = [
                {"id": "R-HSA-168256", "name": "Immune System", "p_value": 0.002, "genes": gene_list[:6]},
                {"id": "R-HSA-1280215", "name": "Cytokine Signaling in Immune system", "p_value": 0.01, "genes": gene_list[3:8]},
                {"id": "R-HSA-1280218", "name": "Adaptive Immune System", "p_value": 0.03, "genes": gene_list[2:5]}
            ]
        else:
            pathways = []
        
        # Example of a real API call:
        # try:
        #     response = requests.post(
        #         f"https://api.pathwaycommons.org/enrichment",
        #         headers={"Authorization": f"Bearer {self.api_keys.get('pathway', '')}"},
        #         json={"genes": gene_list, "database": database}
        #     )
        #     if response.status_code == 200:
        #         pathways = response.json()["results"]
        #     else:
        #         logger.error(f"Error querying Pathway API: {response.text}")
        #         pathways = []
        # except Exception as e:
        #     logger.error(f"Error in Pathway query: {str(e)}")
        #     pathways = []
        
        # Cache results
        results = {
            "query_genes": gene_list,
            "database": database,
            "results": pathways
        }
        
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
    
    def query_disease_association(self, gene_list: List[str]) -> Dict:
        """
        Query disease associations for gene list
        
        Args:
            gene_list: List of gene symbols to query
            
        Returns:
            Dictionary with disease association results
        """
        logger.info(f"Querying disease associations for {len(gene_list)} genes")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"disease_{hash(tuple(sorted(gene_list)))}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Simulate results
        disease_assoc = [
            {"disease": "Type 2 Diabetes", "genes": gene_list[:3], "score": 0.85, "pmids": ["12345678", "23456789"]},
            {"disease": "Rheumatoid Arthritis", "genes": gene_list[2:5], "score": 0.72, "pmids": ["34567890", "45678901"]},
            {"disease": "Alzheimer's Disease", "genes": gene_list[1:4], "score": 0.64, "pmids": ["56789012"]},
            {"disease": "Multiple Sclerosis", "genes": gene_list[3:6], "score": 0.58, "pmids": ["67890123"]},
            {"disease": "Crohn's Disease", "genes": gene_list[4:7], "score": 0.53, "pmids": ["78901234"]}
        ]
        
        # Example of a real API call:
        # try:
        #     response = requests.post(
        #         "https://api.disgenet.org/api/disease_gene",
        #         headers={"Authorization": f"Bearer {self.api_keys.get('disease', '')}"},
        #         json={"genes": gene_list}
        #     )
        #     if response.status_code == 200:
        #         disease_assoc = response.json()["results"]
        #     else:
        #         logger.error(f"Error querying Disease API: {response.text}")
        #         disease_assoc = []
        # except Exception as e:
        #     logger.error(f"Error in Disease query: {str(e)}")
        #     disease_assoc = []
        
        # Cache results
        results = {
            "query_genes": gene_list,
            "results": disease_assoc
        }
        
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
    
    def query_drug_targets(self, gene_list: List[str]) -> Dict:
        """
        Query drug target databases for potential drugs targeting genes
        
        Args:
            gene_list: List of gene symbols to query
            
        Returns:
            Dictionary with drug target results
        """
        logger.info(f"Querying drug targets for {len(gene_list)} genes")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"drug_{hash(tuple(sorted(gene_list)))}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Simulate results
        drug_targets = [
            {"drug": "Methotrexate", "gene": gene_list[0], "interaction_type": "inhibitor", "score": 0.92},
            {"drug": "Adalimumab", "gene": gene_list[1], "interaction_type": "antibody", "score": 0.88},
            {"drug": "Infliximab", "gene": gene_list[2], "interaction_type": "antibody", "score": 0.85},
            {"drug": "Etanercept", "gene": gene_list[1], "interaction_type": "antagonist", "score": 0.82},
            {"drug": "Imatinib", "gene": gene_list[3], "interaction_type": "inhibitor", "score": 0.78}
        ]
        
        # Example of a real API call:
        # try:
        #     response = requests.post(
        #         "https://api.drugbank.com/v1/drugs",
        #         headers={"Authorization": f"Bearer {self.api_keys.get('drug', '')}"},
        #         json={"targets": gene_list}
        #     )
        #     if response.status_code == 200:
        #         drug_targets = response.json()["results"]
        #     else:
        #         logger.error(f"Error querying Drug API: {response.text}")
        #         drug_targets = []
        # except Exception as e:
        #     logger.error(f"Error in Drug query: {str(e)}")
        #     drug_targets = []
        
        # Cache results
        results = {
            "query_genes": gene_list,
            "results": drug_targets
        }
        
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
    
    def query_literature(self, gene_list: List[str], topic: str = None) -> Dict:
        """
        Query literature databases for relevant publications
        
        Args:
            gene_list: List of gene symbols to query
            topic: Optional topic to filter publications
            
        Returns:
            Dictionary with literature results
        """
        logger.info(f"Querying literature for {len(gene_list)} genes")
        
        # Generate cache key
        cache_key = f"literature_{hash(tuple(sorted(gene_list)))}"
        if topic:
            cache_key += f"_{hash(topic)}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Simulate results
        literature = [
            {
                "title": "Single-cell RNA sequencing reveals novel cell differentiation dynamics during human airway epithelium regeneration",
                "authors": "Smith J, Johnson A, et al.",
                "journal": "Nature Communications",
                "year": 2023,
                "pmid": "36123456",
                "genes": [gene_list[0], gene_list[2]],
                "abstract": "This study explores cell differentiation dynamics in airway epithelium regeneration using scRNA-seq."
            },
            {
                "title": "Trajectory analysis of single cells reveals mechanisms of inflammatory response",
                "authors": "Garcia R, Chen Y, et al.",
                "journal": "Cell",
                "year": 2022,
                "pmid": "35123456",
                "genes": [gene_list[1], gene_list[3]],
                "abstract": "Using advanced trajectory analysis, this study identifies key mechanisms of inflammatory response."
            },
            {
                "title": "Characterization of immune cell populations in autoimmune diseases using single-cell transcriptomics",
                "authors": "Wang L, Kim D, et al.",
                "journal": "Science Immunology",
                "year": 2021,
                "pmid": "34123456",
                "genes": [gene_list[0], gene_list[4]],
                "abstract": "This study uses single-cell transcriptomics to characterize immune cell populations in autoimmune diseases."
            }
        ]
        
        # Example of a real API call:
        # try:
        #     query_params = {"genes": ",".join(gene_list)}
        #     if topic:
        #         query_params["topic"] = topic
        #     
        #     response = requests.get(
        #         "https://api.pubmed.ncbi.nlm.nih.gov/v1/search",
        #         headers={"Authorization": f"Bearer {self.api_keys.get('pubmed', '')}"},
        #         params=query_params
        #     )
        #     if response.status_code == 200:
        #         literature = response.json()["results"]
        #     else:
        #         logger.error(f"Error querying Literature API: {response.text}")
        #         literature = []
        # except Exception as e:
        #     logger.error(f"Error in Literature query: {str(e)}")
        #     literature = []
        
        # Cache results
        results = {
            "query_genes": gene_list,
            "topic": topic,
            "results": literature
        }
        
        with open(cache_file, 'w') as f:
            json.dump(results, f)
            
        return results
    
    def enrich_gene_set(self, gene_list: List[str], sources: List[str] = None) -> Dict:
        """
        Enrich a gene set with knowledge from multiple sources
        
        Args:
            gene_list: List of gene symbols to enrich
            sources: List of knowledge sources to query (default: all available)
            
        Returns:
            Dictionary with enrichment results from all sources
        """
        logger.info(f"Enriching gene set with {len(gene_list)} genes")
        
        if not sources:
            sources = list(self.knowledge_sources.keys())
            
        enrichment_results = {}
        
        for source in sources:
            if source in self.knowledge_sources:
                try:
                    enrichment_results[source] = self.knowledge_sources[source](gene_list)
                except Exception as e:
                    logger.error(f"Error enriching genes with {source}: {str(e)}")
                    enrichment_results[source] = {"error": str(e)}
            else:
                logger.warning(f"Unknown knowledge source: {source}")
                
        return enrichment_results
    
    def integrate_with_clusters(self, adata, cluster_key: str = 'leiden', top_n_genes: int = 20) -> Dict:
        """
        Integrate knowledge with cluster information
        
        Args:
            adata: AnnData object
            cluster_key: Key for cluster information in adata.obs
            top_n_genes: Number of top genes per cluster to use
            
        Returns:
            Dictionary with knowledge integration results per cluster
        """
        logger.info(f"Integrating knowledge with cluster information")
        
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key {cluster_key} not found in data")
            
        # Get marker genes for each cluster
        if 'rank_genes_groups' not in adata.uns:
            logger.info("Computing marker genes for clusters")
            import scanpy as sc
            sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
            
        # Extract top marker genes per cluster
        clusters = adata.obs[cluster_key].unique()
        cluster_markers = {}
        
        for cluster in clusters:
            genes = sc.get.rank_genes_groups_names(adata, group=cluster)
            cluster_markers[cluster] = genes[:top_n_genes].tolist()
            
        # Query knowledge sources for each cluster's markers
        cluster_knowledge = {}
        
        for cluster, markers in cluster_markers.items():
            cluster_knowledge[cluster] = self.enrich_gene_set(markers)
            
        return {
            "cluster_markers": cluster_markers,
            "cluster_knowledge": cluster_knowledge
        }
    
    def summarize_cluster_knowledge(self, cluster_knowledge: Dict) -> Dict:
        """
        Summarize knowledge for clusters
        
        Args:
            cluster_knowledge: Dictionary with knowledge integration results per cluster
            
        Returns:
            Dictionary with summarized knowledge per cluster
        """
        logger.info("Summarizing cluster knowledge")
        
        summary = {}
        
        for cluster, knowledge in cluster_knowledge.items():
            cluster_summary = {
                "top_go_terms": [],
                "top_pathways": [],
                "top_diseases": [],
                "potential_drugs": []
            }
            
            # Extract top GO terms
            if "gene_ontology" in knowledge and "results" in knowledge["gene_ontology"]:
                go_terms = knowledge["gene_ontology"]["results"]
                cluster_summary["top_go_terms"] = [
                    {"name": term["name"], "p_value": term["p_value"]} 
                    for term in sorted(go_terms, key=lambda x: x["p_value"])[:3]
                ]
                
            # Extract top pathways
            if "pathway_enrichment" in knowledge and "results" in knowledge["pathway_enrichment"]:
                pathways = knowledge["pathway_enrichment"]["results"]
                cluster_summary["top_pathways"] = [
                    {"name": pathway["name"], "p_value": pathway["p_value"]} 
                    for pathway in sorted(pathways, key=lambda x: x["p_value"])[:3]
                ]
                
            # Extract top diseases
            if "disease_association" in knowledge and "results" in knowledge["disease_association"]:
                diseases = knowledge["disease_association"]["results"]
                cluster_summary["top_diseases"] = [
                    {"name": disease["disease"], "score": disease["score"]} 
                    for disease in sorted(diseases, key=lambda x: x["score"], reverse=True)[:3]
                ]
                
            # Extract potential drugs
            if "drug_target" in knowledge and "results" in knowledge["drug_target"]:
                drugs = knowledge["drug_target"]["results"]
                cluster_summary["potential_drugs"] = [
                    {"name": drug["drug"], "target": drug["gene"], "score": drug["score"]} 
                    for drug in sorted(drugs, key=lambda x: x["score"], reverse=True)[:3]
                ]
                
            summary[cluster] = cluster_summary
            
        return summary