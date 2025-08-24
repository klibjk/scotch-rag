#!/usr/bin/env python3
"""
Test script to simulate web interface integration with RAG system
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag.rag_system import RAGSystem
from dotenv import load_dotenv

def test_web_integration():
    """Test the integration between web interface and RAG system"""
    
    # Load environment
    load_dotenv()
    
    # Initialize RAG system (same as in main_fasthtml.py)
    print("🔧 Initializing RAG system...")
    rag_system = RAGSystem(
        base_storage_dir='./faiss_indexes',
        chunk_size=800,
        chunk_overlap=50,
        max_retrieval_results=7,
    )
    print("✅ RAG system initialized")
    
    # Simulate what the web interface does
    print("\n🤔 Simulating web interface query...")
    
    # Test question (same as what user would type)
    question = "What products are in this dataset?"
    print(f"Question: {question}")
    
    try:
        # This is what the api_ask method does
        print("Calling rag_system.query()...")
        result = rag_system.query(question)
        
        print(f"Result status: {result.get('status')}")
        print(f"Result keys: {list(result.keys())}")
        
        if result["status"] == "success":
            answer = result["answer"]
            print(f"✅ Success! Answer length: {len(answer)}")
            print(f"Answer: {answer}")
            return True
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"❌ Query failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during query: {e}")
        return False

if __name__ == "__main__":
    success = test_web_integration()
    if success:
        print("\n✅ Web integration test completed successfully")
    else:
        print("\n❌ Web integration test failed")
        sys.exit(1)
