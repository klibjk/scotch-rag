"""
Main RAG Application
===================
Terminal interface for the RAG system using the same framework as the reference project.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from rag.rag_system_v1 import get_rag_system
from file_monitor import get_file_monitor

console = Console()

class RAGApplication:
    def __init__(self):
        self.rag_system = get_rag_system()
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        self.file_monitor = get_file_monitor()
    
    def show_welcome(self):
        """Display welcome message"""
        welcome_text = """
🤖 RAG System - Excel Data Query
================================

This system uses the same framework and tools as the reference project:
• LangChain for RAG pipeline
• LlamaParse for Excel processing  
• FAISS for vector storage
• OpenAI embeddings for semantic search
• OpenAI GPT-5-mini for LLM
• Rich terminal interface
• 🔄 Automatic file monitoring

Commands:
• upload <file> - Upload Excel/PDF files
• ask <question> - Ask questions about your data
• ask_web <question> - Ask with web search
• stats - View system statistics
• list - List uploaded files
• reset - Clear all data
• monitor - Start/stop file monitoring
• scan - Scan for new files
• help - Show this help
• quit - Exit the application

🔄 Auto-monitoring: New files in ./data/ will be automatically ingested!
        """
        
        console.print(Panel(welcome_text, title="Welcome", border_style="green"))
    
    def upload_file(self, file_path: str) -> bool:
        """Upload a file to the RAG system"""
        try:
            if not os.path.exists(file_path):
                console.print(f"❌ File not found: {file_path}", style="red")
                return False
            
            console.print(f"📤 Uploading {file_path}...", style="blue")
            
            result = self.rag_system.ingest_file(file_path)
            
            if result["status"] == "success":
                console.print(f"✅ Successfully uploaded {result['filename']}", style="green")
                console.print(f"📊 Created {result['chunks_created']} chunks", style="blue")
                console.print(f"📝 Total characters: {result['total_characters']:,}", style="blue")
                return True
            else:
                console.print(f"❌ Upload failed: {result['error']}", style="red")
                return False
                
        except Exception as e:
            console.print(f"❌ Error uploading file: {e}", style="red")
            return False
    
    def ask_question(self, question: str, use_web_search: bool = False) -> bool:
        """Ask a question to the RAG system"""
        try:
            console.print(f"🤔 Processing question: {question}", style="cyan")
            
            result = self.rag_system.query(question, use_web_search=use_web_search)
            
            if result["status"] == "success":
                # Display answer
                console.print(Panel(
                    result["answer"],
                    title="🤖 Answer",
                    border_style="green"
                ))
                
                # Display quality metrics
                if result.get("quality_metrics"):
                    metrics = result["quality_metrics"]
                    console.print("📊 Quality Metrics:", style="blue")
                    console.print(f"  ⏱️  Query Time: {metrics['query_time']}s", style="dim")
                    console.print(f"  🔍 Retrieval Time: {metrics['retrieval_time']}s", style="dim")
                    console.print(f"  🤖 Generation Time: {metrics['generation_time']}s", style="dim")
                    console.print(f"  📝 Total Tokens: {metrics['total_tokens']}", style="dim")
                    console.print(f"  📚 Sources Found: {metrics['source_count']}", style="dim")
                    console.print(f"  🎯 Relevance Score: {metrics['relevance_score']:.3f}", style="dim")
                    console.print(f"  💪 Confidence Score: {metrics['confidence_score']:.3f}", style="dim")
                    
                    if metrics.get("warnings"):
                        console.print("⚠️  Quality Warnings:", style="yellow")
                        for warning in metrics["warnings"]:
                            console.print(f"  • {warning}", style="yellow")
                
                # Display query analysis
                if result.get("query_analysis"):
                    analysis = result["query_analysis"]
                    console.print("🔍 Query Analysis:", style="blue")
                    console.print(f"  🎯 Intent: {analysis['intent']}", style="dim")
                    console.print(f"  🌍 Language: {analysis['language']}", style="dim")
                    console.print(f"  🔑 Keywords: {', '.join(analysis['keywords'][:5])}", style="dim")
                    
                    if analysis.get("suggestions"):
                        console.print("💡 Suggestions:", style="cyan")
                        for suggestion in analysis["suggestions"]:
                            console.print(f"  • {suggestion}", style="cyan")
                
                # Display sources
                if result["sources"]:
                    console.print("📚 Sources:", style="blue")
                    for i, source in enumerate(result["sources"][:3], 1):
                        console.print(f"  {i}. {source['content']}", style="dim")
                        if source['metadata']:
                            console.print(f"     📄 {source['metadata'].get('filename', 'Unknown')}", style="dim")
                
                # Display web results if available
                if result["web_results"] and result["web_results"].get("results"):
                    console.print("🌐 Web Results:", style="blue")
                    for i, web_result in enumerate(result["web_results"]["results"][:2], 1):
                        console.print(f"  {i}. {web_result.get('title', 'No title')}", style="dim")
                        console.print(f"     {web_result.get('url', 'No URL')}", style="dim")
                
                return True
            else:
                console.print(f"❌ Query failed: {result['error']}", style="red")
                return False
                
        except Exception as e:
            console.print(f"❌ Error processing question: {e}", style="red")
            return False
    
    def show_stats(self):
        """Display system statistics"""
        try:
            stats = self.rag_system.get_stats()
            
            table = Table(title="📊 System Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            if stats["status"] == "active":
                table.add_row("Status", "🟢 Active")
                table.add_row("Total Vectors", str(stats["total_vectors"]))
                table.add_row("Index Size", f"{stats['index_size_mb']} MB")
            else:
                table.add_row("Status", "🔴 No Data")
                table.add_row("Total Vectors", "0")
                table.add_row("Index Size", "0 MB")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"❌ Error getting statistics: {e}", style="red")
    
    def show_quality_metrics(self):
        """Display quality metrics and performance data"""
        try:
            quality_metrics = self.rag_system.get_quality_metrics()
            
            if quality_metrics.get("status") == "no_data":
                console.print("📊 No quality data available yet. Ask some questions first!", style="yellow")
                return
            
            table = Table(title="📊 Quality Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Average Query Time", f"{quality_metrics['avg_query_time']}s")
            table.add_row("Average Confidence", f"{quality_metrics['avg_confidence']:.3f}")
            table.add_row("Average Relevance", f"{quality_metrics['avg_relevance']:.3f}")
            table.add_row("Total Queries", str(quality_metrics['total_queries']))
            table.add_row("Cache Hit Rate", f"{quality_metrics['cache_hit_rate']:.1%}")
            
            console.print(table)
            
            # Show quality recommendations
            recommendations = []
            if quality_metrics['avg_confidence'] < 0.5:
                recommendations.append("Consider using more specific questions")
            if quality_metrics['avg_relevance'] < 0.6:
                recommendations.append("Upload more relevant documents")
            if quality_metrics['avg_query_time'] > 5.0:
                recommendations.append("Consider optimizing document chunking")
            
            if recommendations:
                console.print("💡 Quality Recommendations:", style="cyan")
                for rec in recommendations:
                    console.print(f"  • {rec}", style="cyan")
            
        except Exception as e:
            console.print(f"❌ Error getting quality metrics: {e}", style="red")
    
    def show_fallback_status(self):
        """Display fallback system status"""
        try:
            console.print("🔄 Fallback System Status", style="blue")
            
            # LLM Fallbacks
            console.print("\n🤖 LLM Fallbacks:", style="cyan")
            if self.rag_system.llm_fallbacks:
                for i, fallback in enumerate(self.rag_system.llm_fallbacks, 1):
                    status = "✅" if fallback.get("llm") else "❌"
                    console.print(f"  {i}. {status} {fallback['name']}")
            else:
                console.print("  ⚠️ No LLM fallbacks available")
            
            # Embedding Fallbacks
            console.print("\n🔗 Embedding Fallbacks:", style="cyan")
            if self.rag_system.embedding_fallbacks:
                for i, fallback in enumerate(self.rag_system.embedding_fallbacks, 1):
                    status = "✅" if fallback.get("embeddings") else "❌"
                    console.print(f"  {i}. {status} {fallback['name']}")
            else:
                console.print("  ⚠️ No embedding fallbacks available")
            
            # Vectorstore Fallbacks
            console.print("\n📊 Vectorstore Fallbacks:", style="cyan")
            if self.rag_system.vectorstore_fallbacks:
                for i, fallback in enumerate(self.rag_system.vectorstore_fallbacks, 1):
                    console.print(f"  {i}. ⚙️ {fallback['name']} (requires setup)")
            else:
                console.print("  ⚠️ No vectorstore fallbacks available")
            
            # Cache Fallbacks
            console.print("\n💾 Cache Fallbacks:", style="cyan")
            if self.rag_system.cache_fallbacks:
                for i, fallback in enumerate(self.rag_system.cache_fallbacks, 1):
                    status = "✅" if fallback.get("client") else "❌"
                    console.print(f"  {i}. {status} {fallback['name']}")
            else:
                console.print("  ⚠️ No cache fallbacks available")
            
            # Document Processing Fallbacks
            console.print("\n📄 Document Processing Fallbacks:", style="cyan")
            console.print("  ✅ LlamaParse (primary)")
            console.print("  ✅ Pandas (Excel fallback)")
            console.print("  ✅ PyPDFLoader (PDF fallback)")
            console.print("  ⚙️ Unstructured (general fallback)")
            
        except Exception as e:
            console.print(f"❌ Error getting fallback status: {e}", style="red")
    
    def list_files(self):
        """List uploaded files"""
        try:
            stats = self.rag_system.get_stats()
            
            if stats["status"] == "no_index":
                console.print("📭 No files uploaded yet", style="yellow")
                return
            
            # Get file information from the index
            vectorstore = self.rag_system.load_index()
            if not vectorstore:
                console.print("📭 No files found", style="yellow")
                return
            
            # Extract unique filenames from metadata
            filenames = set()
            for doc in vectorstore.docstore._dict.values():
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('filename', 'Unknown')
                    filenames.add(filename)
            
            if filenames:
                table = Table(title="📋 Uploaded Files")
                table.add_column("Filename", style="cyan")
                table.add_column("Status", style="green")
                
                for filename in sorted(filenames):
                    table.add_row(filename, "✅ Processed")
                
                console.print(table)
            else:
                console.print("📭 No files found", style="yellow")
                
        except Exception as e:
            console.print(f"❌ Error listing files: {e}", style="red")
    
    def reset_system(self):
        """Reset the RAG system"""
        try:
            if Confirm.ask("⚠️ Are you sure you want to reset the system? This will delete all data."):
                self.rag_system.reset()
                console.print("✅ System reset successfully", style="green")
            else:
                console.print("ℹ️ Reset cancelled", style="blue")
                
        except Exception as e:
            console.print(f"❌ Error resetting system: {e}", style="red")
    
    def toggle_monitoring(self):
        """Toggle file monitoring on/off"""
        try:
            if self.file_monitor.is_monitoring:
                self.file_monitor.stop_monitoring()
                console.print("🛑 File monitoring stopped", style="yellow")
            else:
                self.file_monitor.start_monitoring()
                console.print("👀 File monitoring started", style="green")
        except Exception as e:
            console.print(f"❌ Error toggling monitoring: {e}", style="red")
    
    def scan_files(self):
        """Scan for new files in data directory"""
        try:
            console.print("🔍 Scanning for new files...", style="blue")
            self.file_monitor.scan_existing_files()
        except Exception as e:
            console.print(f"❌ Error scanning files: {e}", style="red")
    
    def show_help(self):
        """Display help information"""
        help_text = """
📖 Help - Available Commands
============================

📤 Upload Commands:
• upload <file> - Upload Excel (.xlsx, .xls) or PDF files
• upload data/dataset1.xlsx - Upload specific file

❓ Query Commands:
• ask <question> - Ask questions about your data
• ask "What are the benefits of B3 Plus?"
• ask_web <question> - Ask with web search enhancement

📊 System Commands:
• stats - View system statistics (vectors, index size)
• quality - View quality metrics and performance data
• fallbacks - View fallback system status
• list - List all uploaded files
• reset - Clear all data and start fresh
• monitor - Start/stop automatic file monitoring
• scan - Scan for new files in data directory
• help - Show this help information
• quit - Exit the application

💡 Tips:
• Upload Excel files first before asking questions
• Use specific questions for better answers
• Try ask_web for additional web information
• Check stats to see your data status
• Use 'monitor' to enable automatic file ingestion
• Drop new files in ./data/ directory for auto-processing
        """
        
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def run(self):
        """Main application loop"""
        self.show_welcome()
        
        while True:
            try:
                # Get user input
                command = Prompt.ask("\n[bold cyan]RAG[/bold cyan]").strip()
                
                if not command:
                    continue
                
                # Parse command
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if cmd in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!", style="green")
                    break
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd == 'upload':
                    if not args:
                        file_path = Prompt.ask("Enter file path")
                    else:
                        file_path = args
                    self.upload_file(file_path)
                
                elif cmd == 'ask':
                    if not args:
                        question = Prompt.ask("Enter your question")
                    else:
                        question = args
                    self.ask_question(question)
                
                elif cmd == 'ask_web':
                    if not args:
                        question = Prompt.ask("Enter your question (with web search)")
                    else:
                        question = args
                    self.ask_question(question, use_web_search=True)
                
                elif cmd == 'stats':
                    self.show_stats()
                
                elif cmd == 'quality':
                    self.show_quality_metrics()
                
                elif cmd == 'fallbacks':
                    self.show_fallback_status()
                
                elif cmd == 'list':
                    self.list_files()
                
                elif cmd == 'reset':
                    self.reset_system()
                
                elif cmd == 'monitor':
                    self.toggle_monitoring()
                
                elif cmd == 'scan':
                    self.scan_files()
                
                else:
                    console.print(f"❓ Unknown command: {cmd}. Type 'help' for available commands.", style="yellow")
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="green")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="red")

def main():
    """Main function"""
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        console.print(f"❌ Missing required environment variables: {', '.join(missing_vars)}", style="red")
        console.print("Please set them before running the application.", style="yellow")
        return
    
    # Check for optional LlamaParse
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        console.print("⚠️ LLAMA_CLOUD_API_KEY not set. Excel processing may be limited.", style="yellow")
    
    # Check for optional Tavily
    if not os.getenv("TAVILY_API_KEY"):
        console.print("⚠️ TAVILY_API_KEY not set. Web search will be disabled.", style="yellow")
    
    # Use OpenAI embeddings only
    console.print("🔗 Using OpenAI embeddings (paid, high quality)", style="blue")
    
    # Run the application
    app = RAGApplication()
    
    # Start file monitoring automatically
    app.file_monitor.start_monitoring()
    
    app.run()

if __name__ == "__main__":
    main()
