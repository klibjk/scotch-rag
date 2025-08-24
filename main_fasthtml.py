"""
Scotch-RAG Main Application

This is the main entry point for the Scotch-RAG application.
It initializes the RAG system and starts the web interface.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from rag.rag_system_llamaparse import RAGSystemLlamaParse
from frontend import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scotch_rag.log")],
)

logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from config files."""
    # Try to load from config/environment.local first
    local_env = Path("config/environment.local")
    if local_env.exists():
        load_dotenv(local_env)
        logger.info("Loaded environment from config/environment.local")
    else:
        # Fallback to .env file
        load_dotenv()
        logger.info("Loaded environment from .env file")


def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error(
            "Please set these variables in config/environment.local or .env file"
        )
        return False

    logger.info("Environment validation passed")
    return True


def create_rag_config():
    """Create RAG configuration from environment variables."""
    return {
        "base_storage_dir": os.getenv("RAG_INDEX_PATH", "./faiss_indexes"),
        "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "250")),
        "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        "max_retrieval_results": int(os.getenv("RAG_MAX_RETRIEVAL", "7")),
    }


def main():
    """Main application entry point."""
    logger.info("Starting Scotch-RAG application...")

    # Load environment variables
    load_environment()

    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    try:
        # Create RAG configuration
        config = create_rag_config()
        logger.info(
            f"RAG configuration created with chunk_size: {config['chunk_size']}"
        )

        # Initialize RAG systems
        logger.info("Initializing RAG systems...")

        # Initialize pandas-based RAG system
        rag_system_pandas = RAGSystem(**config)
        logger.info("Pandas RAG system initialized successfully")

        # Initialize LlamaParse-based RAG system
        rag_system_llamaparse = RAGSystemLlamaParse(
            base_storage_dir="./faiss_indexes_llamaparse",
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            max_retrieval_results=config["max_retrieval_results"],
        )
        logger.info("LlamaParse RAG system initialized successfully")

        # Auto-ingest test file into both RAG systems
        test_file_path = "rag/data/scotch_product_catalog.xlsx"
        if os.path.exists(test_file_path):
            logger.info(f"Auto-ingesting test file: {test_file_path}")

            # Ingest into Pandas RAG system
            pandas_result = rag_system_pandas.ingest_file(test_file_path)
            if pandas_result.get("status") == "success":
                logger.info(
                    f"✅ Pandas RAG: Ingested {pandas_result.get('chunks_created', 0)} chunks"
                )
            else:
                logger.warning(
                    f"⚠️ Pandas RAG ingestion failed: {pandas_result.get('error', 'Unknown error')}"
                )

            # Ingest into LlamaParse RAG system
            llamaparse_result = rag_system_llamaparse.ingest_file(test_file_path)
            if llamaparse_result.get("status") == "success":
                logger.info(
                    f"✅ LlamaParse RAG: Ingested {llamaparse_result.get('chunks_created', 0)} chunks"
                )
            else:
                logger.warning(
                    f"⚠️ LlamaParse RAG ingestion failed: {llamaparse_result.get('error', 'Unknown error')}"
                )
        else:
            logger.warning(f"Test file not found: {test_file_path}")

        # Create web application
        logger.info("Creating web application...")
        app = create_app(rag_system_pandas, rag_system_llamaparse)
        logger.info("Web application created successfully")

        # Get server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        debug = os.getenv("DEBUG", "True").lower() == "true"

        logger.info(f"Starting server on {host}:{port} (debug={debug})")

        # Start the application
        app.run(host=host, port=port, debug=debug)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
