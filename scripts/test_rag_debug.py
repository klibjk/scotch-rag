#!/usr/bin/env python3
"""
Debug script to test RAG system functionality
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from dotenv import load_dotenv


def test_rag_system():
    """Test the RAG system directly"""

    # Load environment
    load_dotenv()

    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return False

    print(f"✅ OpenAI API key found: {api_key[:10]}...")

    # Initialize RAG system
    print("🔧 Initializing RAG system...")
    try:
        rag_system = RAGSystem(
            base_storage_dir="./faiss_indexes",
            chunk_size=800,
            chunk_overlap=50,
            max_retrieval_results=7,
        )
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

    # Check system status
    print("\n📊 Checking system status...")
    try:
        stats = rag_system.get_stats()
        print(f"Status: {stats.get('status')}")
        print(f"Total vectors: {stats.get('total_vectors', 0)}")
        print(f"Index size: {stats.get('index_size_mb', 0)} MB")
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return False

    # Test a simple query
    print("\n🤔 Testing query...")
    try:
        result = rag_system.query("Hello, this is a test question")
        print(f"Query result status: {result.get('status')}")
        print(f"Query result keys: {list(result.keys())}")

        if result.get("status") == "success":
            print(f"✅ Query successful!")
            print(f"Answer: {result.get('answer', 'No answer')}")
        else:
            print(f"❌ Query failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Query failed with exception: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_rag_system()
    if success:
        print("\n✅ RAG system test completed successfully")
    else:
        print("\n❌ RAG system test failed")
        sys.exit(1)
