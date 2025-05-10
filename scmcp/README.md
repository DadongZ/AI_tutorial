# Single Cell Model Context Protocol (MCP)

This project implements a Model Context Protocol (MCP) for single-cell RNA-seq data analysis. The system integrates traditional analysis techniques with deep learning and LLM-based insights.

## Key Components

1. **Data Sources**: Integration with scRNA-seq, ATAC-seq, spatial transcriptomics, and more
2. **MCP Core**: 
   - Data transformation rules
   - Context integration
   - Prompt engineering
3. **Deep Insight Discovery Engine**: Pattern recognition between conditions
4. **Knowledge Base**: Integration with pathway databases, gene ontology, literature
5. **Process/Analysis Tools**: Monocle3, GSEA, Network analysis, etc.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sc-mcp.git
cd sc-mcp

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Basic usage
python main.py --data path/to/your/data.h5ad --condition condition_column_name

# Interactive mode
python main.py
```

### Command Line Interface
The MCP provides a command-line interface with the following commands:

- load: Load single-cell data file
- preprocess: Preprocess loaded data
- trajectory: Run trajectory analysis using Monocle3
- insights: Discover deep insights in the data
- visualize: Visualize results
- summary: Generate a summary of the analysis results
- search: Search knowledge base
- query: Process a natural language query
- save: Save current analysis state
- exit or quit: Exit the program

### Example Workflow
- Load data: load data/my_single_cell_data.h5ad
- Preprocess: preprocess
- Run trajectory analysis: trajectory
- Discover insights: insights condition=treatment
- Visualize results: visualize umap color=cell_type
- Generate summary: summary

### File Structure
- main.py: Main entry point
- mcp_sc_analysis.py: Core MCP implementation
- interface.py: User interface
- monocle3.py: Trajectory analysis
- knowledge_integration.py: Knowledge base integration
- deep_insight.py: Deep insight discovery engine
- visualization.py: Visualization utilities
- llm_integration.py: LLM integration for context and prompt engineering
- config.py: Configuration management

### Requirements
- Python 3.8+
- TensorFlow 2.6+
- Scanpy 1.8+
- Pandas 1.3+
- NumPy 1.20+