# llm_integration.py

import os
import json
import requests
from typing import Dict, List, Any, Optional

class LLMIntegration:
    def __init__(self, model="gpt-4", api_key=None):
        """
        Initialize LLM integration for MCP
        
        Parameters:
        -----------
        model : str
            LLM model to use
        api_key : str
            API key for LLM access
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.context = []
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        """Load prompt templates"""
        templates = {
            "data_transform": """
                As an expert in single cell analysis, transform the following data:
                
                {data_description}
                
                Transform this data to:
                {transformation_goal}
                
                Consider the following context:
                {context}
            """,
            
            "context_integration": """
                Integrate the following context information into your analysis:
                
                Cell data:
                {cell_data}
                
                Experimental conditions:
                {conditions}
                
                Knowledge base facts:
                {knowledge_facts}
                
                Based on this information, provide insights about:
                {query}
            """,
            
            "pattern_discovery": """
                Analyze the following single cell data patterns:
                
                Differential genes:
                {diff_genes}
                
                Trajectory information:
                {trajectory_info}
                
                Identify biologically meaningful patterns related to:
                {biological_process}
            """,
            
            "query_analysis": """
                The user has asked the following query about single cell data:
                
                {user_query}
                
                The available data has these characteristics:
                {data_characteristics}
                
                What analysis steps should be performed to answer this query?
                Consider preprocessing, trajectory analysis, differential expression, and knowledge integration.
            """
        }
        return templates
    
    def generate_prompt(self, template_name: str, **kwargs):
        """
        Generate a prompt using a template
        
        Parameters:
        -----------
        template_name : str
            Name of the template to use
        **kwargs : dict
            Variables to fill in the template
        """
        if template_name not in self.prompt_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.prompt_templates[template_name]
        
        # Fill in template variables
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            template = template.replace(placeholder, str(value))
        
        return template
    
    def query_llm(self, prompt: str, temperature=0.7, max_tokens=1000):
        """
        Query the LLM with a prompt
        
        Parameters:
        -----------
        prompt : str
            Prompt to send to the LLM
        temperature : float
            Temperature for generation
        max_tokens : int
            Maximum number of tokens to generate
        """
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENAI_API_KEY environment variable or pass api_key.")
        
        # Example using OpenAI API - adjust based on the actual LLM API you're using
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "No response from LLM."
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def analyze_query(self, user_query: str, data_characteristics: Dict[str, Any]):
        """
        Analyze a user query to determine appropriate analysis steps
        
        Parameters:
        -----------
        user_query : str
            Natural language query from the user
        data_characteristics : dict
            Characteristics of the available data
        """
        prompt = self.generate_prompt(
            "query_analysis",
            user_query=user_query,
            data_characteristics=json.dumps(data_characteristics, indent=2)
        )
        
        return self.query_llm(prompt)
    
    def transform_data(self, data_description: str, transformation_goal: str, context: str = ""):
        """
        Generate instructions for data transformation
        
        Parameters:
        -----------
        data_description : str
            Description of the data to transform
        transformation_goal : str
            Goal of the transformation
        context : str
            Additional context information
        """
        prompt = self.generate_prompt(
            "data_transform",
            data_description=data_description,
            transformation_goal=transformation_goal,
            context=context
        )
        
        return self.query_llm(prompt)
    
    def integrate_context(self, cell_data: str, conditions: str, knowledge_facts: str, query: str):
        """
        Integrate context information for analysis
        
        Parameters:
        -----------
        cell_data : str
            Description of cell data
        conditions : str
            Description of experimental conditions
        knowledge_facts : str
            Relevant knowledge base facts
        query : str
            User query to address
        """
        prompt = self.generate_prompt(
            "context_integration",
            cell_data=cell_data,
            conditions=conditions,
            knowledge_facts=knowledge_facts,
            query=query
        )
        
        return self.query_llm(prompt)
    
    def discover_patterns(self, diff_genes: str, trajectory_info: str, biological_process: str):
        """
        Discover patterns in single cell data
        
        Parameters:
        -----------
        diff_genes : str
            Information about differential genes
        trajectory_info : str
            Information about trajectory analysis
        biological_process : str
            Biological process of interest
        """
        prompt = self.generate_prompt(
            "pattern_discovery",
            diff_genes=diff_genes,
            trajectory_info=trajectory_info,
            biological_process=biological_process
        )
        
        return self.query_llm(prompt)