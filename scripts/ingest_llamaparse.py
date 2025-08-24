#!/usr/bin/env python3
"""
Script to ingest files into the LlamaParse RAG system for testing.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.rag_system_llamaparse import RAGSystemLlamaParse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def main():
    """Main function to ingest files into LlamaParse RAG system."""
    
    # Load environment variables
    load_dotenv()
    
    # Test file to ingest
    test_file = "rag/data/scotch_product_catalog.xlsx"
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        # Initialize LlamaParse RAG system
        logger.info("Initializing LlamaParse RAG system...")
        rag_system = RAGSystemLlamaParse(
            base_storage_dir="./faiss_indexes_llamaparse",
            chunk_size=800,
            chunk_overlap=50,
            max_retrieval_results=7
        )
        logger.info("LlamaParse RAG system initialized successfully")
        
        # Ingest the file
        logger.info(f"Ingesting file: {test_file}")
        result = rag_system.ingest_file(test_file)
        
        if result:
            logger.info("✅ File ingested successfully into LlamaParse system!")
            
            # Test a query
            logger.info("Testing query...")
            test_question = "How many products are there?"
            answer = rag_system.query(test_question)
            
            if answer and "error" not in answer.lower():
                logger.info(f"✅ Query successful! Answer: {answer[:100]}...")
            else:
                logger.warning(f"⚠️ Query failed: {answer}")
            
            return True
        else:
            logger.error("❌ Ingestion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during ingestion: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ LlamaParse ingestion test completed successfully")
    else:
        logger.error("❌ LlamaParse ingestion test failed")
        sys.exit(1)
