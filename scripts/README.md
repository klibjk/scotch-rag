# Scripts Directory

This directory contains utility scripts for testing, ingestion, and development of the Scotch-RAG system.

## Scripts Overview

### Ingestion Scripts
- **`ingest_file.py`** - Generic file ingestion script with command-line arguments
- **`ingest_llamaparse.py`** - Dedicated script to ingest files into the LlamaParse RAG system

### Testing Scripts
- **`test_ingest.py`** - Test file ingestion functionality
- **`test_rag_debug.py`** - Debug RAG system functionality
- **`test_dual_rag.py`** - Compare performance between Pandas and LlamaParse RAG systems
- **`test_api.py`** - Test API endpoints
- **`test_web_integration.py`** - Test web interface integration

## Usage

Most scripts can be run directly with Python:

```bash
python3 scripts/ingest_llamaparse.py
python3 scripts/test_dual_rag.py
```

## Notes

- These scripts are primarily for development and testing purposes
- The main application uses `main_fasthtml.py` in the root directory
- Some scripts may require specific environment variables or API keys to be set
