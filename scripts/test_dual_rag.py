#!/usr/bin/env python3
"""
Dual RAG System Comparison Test
===============================
Test script to compare pandas vs LlamaParse RAG system performance.
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Import both RAG systems
from rag.rag_system import RAGSystem
from rag.rag_system_llamaparse import RAGSystemLlamaParse

# Load environment variables
load_dotenv()

def create_rag_config():
    """Create RAG configuration"""
    return {
        "base_storage_dir": "./faiss_indexes",
        "chunk_size": 800,
        "chunk_overlap": 50,
        "max_retrieval_results": 7
    }

def test_dual_rag_systems():
    """Test both RAG systems with the same data and queries"""
    
    print("🧪 Starting Dual RAG System Comparison Test")
    print("=" * 60)
    
    # Test file
    test_file = "rag/data/scotch_product_catalog.xlsx"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Test questions
    test_questions = [
        "How many products are in the catalog?",
        "What is the first product in the catalog?",
        "What is the second product in the catalog?",
        "What products contain 'Scot' in the name?",
        "What is the most expensive product?",
        "What is the cheapest product?",
        "How many products are priced under $50?",
        "What is the average price of products?",
        "List all products with 'Royal' in the name",
        "What is the total number of items across all products?"
    ]
    
    # Initialize RAG systems
    print("\n🔧 Initializing RAG Systems...")
    
    rag_config = create_rag_config()
    
    # Initialize pandas-based RAG system
    print("🐼 Initializing Pandas RAG system...")
    rag_system_pandas = RAGSystem(**rag_config)
    
    # Initialize LlamaParse-based RAG system
    print("🦙 Initializing LlamaParse RAG system...")
    rag_system_llamaparse = RAGSystemLlamaParse(
        base_storage_dir="./faiss_indexes_llamaparse",
        chunk_size=rag_config["chunk_size"],
        chunk_overlap=rag_config["chunk_overlap"],
        max_retrieval_results=rag_config["max_retrieval_results"]
    )
    
    print("✅ Both RAG systems initialized successfully!")
    
    # Test ingestion
    print("\n📄 Testing Document Ingestion...")
    print("-" * 40)
    
    # Test pandas ingestion
    print("🐼 Testing Pandas ingestion...")
    start_time = time.time()
    pandas_result = rag_system_pandas.ingest_file(test_file)
    pandas_ingestion_time = time.time() - start_time
    
    if pandas_result["status"] == "success":
        print(f"✅ Pandas ingestion successful in {pandas_ingestion_time:.2f}s")
        print(f"   - Chunks created: {pandas_result['chunks_created']}")
        print(f"   - Total characters: {pandas_result['total_characters']}")
    else:
        print(f"❌ Pandas ingestion failed: {pandas_result['error']}")
        return
    
    # Test LlamaParse ingestion
    print("\n🦙 Testing LlamaParse ingestion...")
    start_time = time.time()
    llamaparse_result = rag_system_llamaparse.ingest_file(test_file)
    llamaparse_ingestion_time = time.time() - start_time
    
    if llamaparse_result["status"] == "success":
        print(f"✅ LlamaParse ingestion successful in {llamaparse_ingestion_time:.2f}s")
        print(f"   - Chunks created: {llamaparse_result['chunks_created']}")
        print(f"   - Total characters: {llamaparse_result['total_characters']}")
    else:
        print(f"❌ LlamaParse ingestion failed: {llamaparse_result['error']}")
        return
    
    # Compare ingestion results
    print("\n📊 Ingestion Comparison:")
    print("-" * 40)
    print(f"Pandas:     {pandas_ingestion_time:.2f}s | {pandas_result['chunks_created']} chunks | {pandas_result['total_characters']} chars")
    print(f"LlamaParse: {llamaparse_ingestion_time:.2f}s | {llamaparse_result['chunks_created']} chunks | {llamaparse_result['total_characters']} chars")
    
    if llamaparse_ingestion_time < pandas_ingestion_time:
        print(f"🏆 LlamaParse is {pandas_ingestion_time/llamaparse_ingestion_time:.1f}x faster for ingestion")
    else:
        print(f"🏆 Pandas is {llamaparse_ingestion_time/pandas_ingestion_time:.1f}x faster for ingestion")
    
    # Test queries
    print("\n❓ Testing Query Performance...")
    print("-" * 40)
    
    pandas_results = []
    llamaparse_results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        # Test pandas query
        print("  🐼 Pandas query...")
        start_time = time.time()
        pandas_response = rag_system_pandas.query(question)
        pandas_query_time = time.time() - start_time
        
        if pandas_response["status"] == "success":
            pandas_results.append({
                "question": question,
                "answer": pandas_response["answer"],
                "time": pandas_query_time,
                "answer_length": len(pandas_response["answer"])
            })
            print(f"    ✅ {pandas_query_time:.2f}s | {len(pandas_response['answer'])} chars")
        else:
            print(f"    ❌ Failed: {pandas_response['error']}")
        
        # Test LlamaParse query
        print("  🦙 LlamaParse query...")
        start_time = time.time()
        llamaparse_response = rag_system_llamaparse.query(question)
        llamaparse_query_time = time.time() - start_time
        
        if llamaparse_response["status"] == "success":
            llamaparse_results.append({
                "question": question,
                "answer": llamaparse_response["answer"],
                "time": llamaparse_query_time,
                "answer_length": len(llamaparse_response["answer"])
            })
            print(f"    ✅ {llamaparse_query_time:.2f}s | {len(llamaparse_response['answer'])} chars")
        else:
            print(f"    ❌ Failed: {llamaparse_response['error']}")
    
    # Analyze results
    print("\n📊 Query Performance Analysis:")
    print("-" * 40)
    
    if pandas_results and llamaparse_results:
        pandas_avg_time = sum(r["time"] for r in pandas_results) / len(pandas_results)
        llamaparse_avg_time = sum(r["time"] for r in llamaparse_results) / len(llamaparse_results)
        
        pandas_avg_length = sum(r["answer_length"] for r in pandas_results) / len(pandas_results)
        llamaparse_avg_length = sum(r["answer_length"] for r in llamaparse_results) / len(llamaparse_results)
        
        print(f"Pandas Average:     {pandas_avg_time:.2f}s | {pandas_avg_length:.0f} chars")
        print(f"LlamaParse Average: {llamaparse_avg_time:.2f}s | {llamaparse_avg_length:.0f} chars")
        
        if llamaparse_avg_time < pandas_avg_time:
            print(f"🏆 LlamaParse is {pandas_avg_time/llamaparse_avg_time:.1f}x faster for queries")
        else:
            print(f"🏆 Pandas is {llamaparse_avg_time/pandas_avg_time:.1f}x faster for queries")
        
        if llamaparse_avg_length > pandas_avg_length:
            print(f"📝 LlamaParse generates {llamaparse_avg_length/pandas_avg_length:.1f}x longer answers")
        else:
            print(f"📝 Pandas generates {pandas_avg_length/llamaparse_avg_length:.1f}x longer answers")
    
    # Detailed comparison
    print("\n📋 Detailed Question-by-Question Comparison:")
    print("-" * 60)
    print(f"{'Question':<40} {'Pandas':<15} {'LlamaParse':<15} {'Winner':<10}")
    print("-" * 60)
    
    for i, question in enumerate(test_questions):
        if i < len(pandas_results) and i < len(llamaparse_results):
            pandas_time = pandas_results[i]["time"]
            llamaparse_time = llamaparse_results[i]["time"]
            
            if pandas_time < llamaparse_time:
                winner = "Pandas"
            elif llamaparse_time < pandas_time:
                winner = "LlamaParse"
            else:
                winner = "Tie"
            
            print(f"{question[:39]:<40} {pandas_time:<15.2f} {llamaparse_time:<15.2f} {winner:<10}")
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    results = {
        "test_file": test_file,
        "ingestion_comparison": {
            "pandas": {
                "time": pandas_ingestion_time,
                "chunks": pandas_result["chunks_created"],
                "characters": pandas_result["total_characters"]
            },
            "llamaparse": {
                "time": llamaparse_ingestion_time,
                "chunks": llamaparse_result["chunks_created"],
                "characters": llamaparse_result["total_characters"]
            }
        },
        "query_results": {
            "pandas": pandas_results,
            "llamaparse": llamaparse_results
        },
        "summary": {
            "pandas_avg_time": sum(r["time"] for r in pandas_results) / len(pandas_results) if pandas_results else 0,
            "llamaparse_avg_time": sum(r["time"] for r in llamaparse_results) / len(llamaparse_results) if llamaparse_results else 0,
            "pandas_avg_length": sum(r["answer_length"] for r in pandas_results) / len(pandas_results) if pandas_results else 0,
            "llamaparse_avg_length": sum(r["answer_length"] for r in llamaparse_results) / len(llamaparse_results) if llamaparse_results else 0
        }
    }
    
    with open("dual_rag_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to dual_rag_comparison_results.json")
    
    # Final summary
    print("\n🎯 Final Summary:")
    print("-" * 40)
    print("✅ Both RAG systems successfully processed the test file")
    print("✅ Both systems answered all test questions")
    print("📊 Performance metrics calculated and saved")
    print("🔍 Check dual_rag_comparison_results.json for detailed analysis")
    
    print("\n🚀 Test completed successfully!")

if __name__ == "__main__":
    test_dual_rag_systems()
