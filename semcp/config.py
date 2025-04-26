# config.py

import os
import json
from pathlib import Path

class Config:
    def __init__(self, config_file=None):
        """Initialize configuration"""
        self.config_file = config_file or os.path.join(os.path.dirname(__file__), 'config.json')
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {str(e)}")
                return self._default_config()
        else:
            # Create default config
            config = self._default_config()
            self._save_config(config)
            return config
    
    def _default_config(self):
        """Return default configuration"""
        return {
            "data": {
                "default_dir": "./data",
                "metadata_fields": ["condition", "cell_type", "batch"]
            },
            "preprocessing": {
                "min_genes": 200,
                "min_cells": 3,
                "target_sum": 1e4,
                "n_hvgs": 2000
            },
            "trajectory": {
                "use_genes": "hvg",
                "n_neighbors": 15,
                "n_pcs": 50
            },
            "deep_insight": {
                "model": {
                    "hidden_dims": [512, 256],
                    "encoding_dim": 128,
                    "epochs": 50,
                    "batch_size": 64,
                    "learning_rate": 0.001
                },
                "pattern_analysis": {
                    "top_n_genes": 100,
                    "integration_threshold": 0.05
                }
            },
            "visualization": {
                "umap": {
                    "n_neighbors": 15,
                    "min_dist": 0.5
                },
                "output_format": "png",
                "dpi": 300
            },
            "knowledge_base": {
                "pathway_db": "kegg",
                "go_db": "GO_Biological_Process_2021",
                "disease_db": "DisGeNET"
            },
            "output": {
                "dir": "./output",
                "save_plots": True,
                "save_data": True
            }
        }
    
    def _save_config(self, config):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config file: {str(e)}")
    
    def get(self, section, key=None):
        """Get configuration value"""
        if section not in self.config:
            return None
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, None)
    
    def set(self, section, key, value):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        self._save_config(self.config)
    
    def update(self, updates):
        """Update multiple configuration values"""
        for section, section_updates in updates.items():
            if section not in self.config:
                self.config[section] = {}
            
            for key, value in section_updates.items():
                self.config[section][key] = value
        
        self._save_config(self.config)