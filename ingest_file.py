#!/usr/bin/env python3
"""
Flexible file ingestion script for Scotch-RAG
Usage: python3 ingest_file.py <file_path>
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from dotenv import load_dotenv


def ingest_file(file_path: str):
    """Ingest a file into the RAG system"""

    # Load environment
    load_dotenv()

    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Initialize RAG system
    print("🔧 Initializing RAG system...")
    rag_system = RAGSystem(
        base_storage_dir="./faiss_indexes",
        chunk_size=800,
        chunk_overlap=50,
        max_retrieval_results=7,
    )
    print("✅ RAG system initialized")

    # Check initial status
    print("\n📊 Initial system status:")
    stats = rag_system.get_stats()
    print(f"Status: {stats.get('status')}")
    print(f"Total vectors: {stats.get('total_vectors', 0)}")

    print(f"\n📄 Ingesting file: {file_path}")

    try:
        # Ingest the file
        result = rag_system.ingest_file(file_path)

        print(f"Ingestion result: {result}")

        if result.get("status") == "success":
            print("✅ File ingested successfully!")

            # Check status after ingestion
            print("\n📊 System status after ingestion:")
            stats = rag_system.get_stats()
            print(f"Status: {stats.get('status')}")
            print(f"Total vectors: {stats.get('total_vectors', 0)}")
            return True
        else:
            print(f"❌ File ingestion failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Exception during ingestion: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ingest_file.py <file_path>")
        print("Example: python3 ingest_file.py rag/data/dataset2.xlsx")
        sys.exit(1)

    file_path = sys.argv[1]
    success = ingest_file(file_path)

    if success:
        print("\n✅ File ingestion completed successfully")
    else:
        print("\n❌ File ingestion failed")
        sys.exit(1)
