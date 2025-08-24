#!/usr/bin/env python3
"""
Test script to ingest a file into the RAG system
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from dotenv import load_dotenv


def test_file_ingestion():
    """Test ingesting a file into the RAG system"""

    # Load environment
    load_dotenv()

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

    # Test file to ingest - change this to ingest different files
    test_file = "rag/data/scotch_product_catalog.xlsx"  # Change this path to ingest different files

    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"\n📄 Ingesting file: {test_file}")

    try:
        # Ingest the file
        result = rag_system.ingest_file(test_file)

        print(f"Ingestion result: {result}")

        if result.get("status") == "success":
            print("✅ File ingested successfully!")

            # Check status after ingestion
            print("\n📊 System status after ingestion:")
            stats = rag_system.get_stats()
            print(f"Status: {stats.get('status')}")
            print(f"Total vectors: {stats.get('total_vectors', 0)}")

            # Test a query
            print("\n🤔 Testing query after ingestion...")
            query_result = rag_system.query("What is this dataset about?")

            print(f"Query status: {query_result.get('status')}")
            if query_result.get("status") == "success":
                print(f"✅ Query successful!")
                print(f"Answer: {query_result.get('answer', 'No answer')}")
                return True
            else:
                print(f"❌ Query failed: {query_result.get('error', 'Unknown error')}")
                return False

        else:
            print(f"❌ File ingestion failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Exception during ingestion: {e}")
        return False


if __name__ == "__main__":
    success = test_file_ingestion()
    if success:
        print("\n✅ File ingestion test completed successfully")
    else:
        print("\n❌ File ingestion test failed")
        sys.exit(1)
