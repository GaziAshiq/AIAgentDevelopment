# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Agent Development repository containing educational notebooks and code examples for building AI agents. The project explores multiple frameworks including LangChain, LangGraph, and OpenAI Agents SDK.

## Dependencies and Environment Setup

This project uses `uv` for dependency management with Python 3.13+. 

### Common Commands

```bash
# Install/sync dependencies
uv sync

# Update dependencies (uncomment line 32 in pyproject.toml)
uv sync upgrade

# Run Jupyter notebooks
jupyter lab
# or
jupyter notebook

# Run the main Python script
python main.py
# or
uv run python main.py
```

## Project Architecture

### Directory Structure

- **Agentic AI Engineering/**: Advanced agent development topics
  - `1_Foundations/`: Basic AI workflows, multi-LLM APIs, web chatbots, function calling
  - `2_OpenAI_Agents_SDK/`: OpenAI Agents SDK implementation examples
  - `LangGraph/`: LangGraph framework fundamentals
- **Learning_Notes/**: Educational notebooks covering core concepts
- **M1-M5 directories**: Module-based learning structure covering FAQ bots, LLM prompting, memory systems, and news aggregation
- **main.py**: Simple entry point script

### Key Technologies

The project integrates multiple AI/ML frameworks:
- **OpenAI**: Primary LLM provider (`openai`, `openai-agents`)  
- **Google AI**: Gemini integration (`google-genai`, `langchain-google-genai`)
- **Anthropic**: Claude integration (`anthropic`)
- **LangChain**: Agent framework (`langchain`, `langchain-community`, `langchain-core`)
- **LangGraph**: Workflow orchestration (`langgraph`)
- **ChromaDB**: Vector database for embeddings (`chromadb`, `langchain-chroma`)
- **Additional**: Gradio for UIs, PyMuPDF/PyPDF for document processing, Docling for document parsing

### Notebook Organization

Most work is done in Jupyter notebooks (.ipynb files) organized by learning modules:
- Each major directory has an `__init__.py` file for Python package structure
- Notebooks demonstrate practical implementations of AI agent concepts
- Examples progress from basic FAQ bots to complex multi-agent systems

## Development Notes

- The project uses `.env` files for API keys and configuration
- Vector databases and embeddings are used extensively for context-aware agents  
- Multiple LLM providers are supported for comparison and flexibility
- Document processing capabilities support PDF and other formats via PyMuPDF/Docling