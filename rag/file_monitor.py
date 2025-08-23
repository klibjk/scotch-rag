"""
File Monitor for Automatic Ingestion
===================================
Automatically monitors the data directory and ingests new files.
Based on the reference project's file monitoring approach.
"""

import os
import time
import threading
from pathlib import Path
from typing import Set, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from rich.console import Console
from rich.panel import Panel

from rag_system_v1 import get_rag_system

console = Console()

class DataFileHandler(FileSystemEventHandler):
    """Handler for file system events in the data directory"""
    
    def __init__(self, rag_system, data_dir: Path):
        self.rag_system = rag_system
        self.data_dir = data_dir
        self.processed_files: Set[str] = set()
        self.ignored_patterns = {'.DS_Store', 'Thumbs.db', '~$', 'QA.xlsx'}
        
        # Load already processed files
        self._load_processed_files()
    
    def _load_processed_files(self):
        """Load list of already processed files"""
        try:
            # Check existing index to see what's already processed
            vectorstore = self.rag_system.load_index()
            if vectorstore:
                for doc in vectorstore.docstore._dict.values():
                    if hasattr(doc, 'metadata') and doc.metadata:
                        filename = doc.metadata.get('filename', '')
                        if filename:
                            self.processed_files.add(filename)
                
                console.print(f"📋 Loaded {len(self.processed_files)} previously processed files", style="blue")
        except Exception as e:
            console.print(f"⚠️ Could not load processed files list: {e}", style="yellow")
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        filename = os.path.basename(file_path)
        
        # Skip ignored patterns
        for pattern in self.ignored_patterns:
            if pattern in filename:
                return False
        
        # Skip already processed files
        if filename in self.processed_files:
            return False
        
        # Only process Excel and PDF files
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.xlsx', '.xls', '.pdf']:
            return False
        
        return True
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self._should_process_file(file_path):
            console.print(f"🆕 New file detected: {os.path.basename(file_path)}", style="green")
            
            # Wait a moment for file to be fully written
            time.sleep(1)
            
            # Process the file
            self._process_new_file(file_path)
    
    def on_moved(self, event):
        """Handle file move events (like when files are copied)"""
        if event.is_directory:
            return
        
        file_path = event.dest_path
        if self._should_process_file(file_path):
            console.print(f"📁 File moved to data directory: {os.path.basename(file_path)}", style="green")
            
            # Wait a moment for file to be fully written
            time.sleep(1)
            
            # Process the file
            self._process_new_file(file_path)
    
    def _process_new_file(self, file_path: str):
        """Process a new file"""
        try:
            filename = os.path.basename(file_path)
            console.print(f"📤 Auto-ingesting: {filename}", style="blue")
            
            result = self.rag_system.ingest_file(file_path)
            
            if result["status"] == "success":
                console.print(f"✅ Auto-ingested: {filename}", style="green")
                console.print(f"   📊 Created {result['chunks_created']} chunks", style="cyan")
                self.processed_files.add(filename)
            else:
                console.print(f"❌ Auto-ingest failed for {filename}: {result['error']}", style="red")
                
        except Exception as e:
            console.print(f"❌ Error auto-ingesting {os.path.basename(file_path)}: {e}", style="red")

class FileMonitor:
    """File monitor for automatic ingestion"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rag_system = get_rag_system()
        self.observer = None
        self.handler = None
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start monitoring the data directory"""
        if self.is_monitoring:
            console.print("⚠️ File monitoring is already active", style="yellow")
            return
        
        try:
            console.print(f"👀 Starting file monitor for: {self.data_dir}", style="blue")
            
            # Create handler and observer
            self.handler = DataFileHandler(self.rag_system, self.data_dir)
            self.observer = Observer()
            self.observer.schedule(self.handler, str(self.data_dir), recursive=False)
            
            # Start monitoring
            self.observer.start()
            self.is_monitoring = True
            
            console.print("✅ File monitoring started", style="green")
            console.print("📁 Drop new Excel/PDF files in the data directory to auto-ingest", style="cyan")
            
        except Exception as e:
            console.print(f"❌ Failed to start file monitoring: {e}", style="red")
    
    def stop_monitoring(self):
        """Stop monitoring the data directory"""
        if not self.is_monitoring:
            return
        
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            self.is_monitoring = False
            console.print("🛑 File monitoring stopped", style="yellow")
            
        except Exception as e:
            console.print(f"❌ Error stopping file monitoring: {e}", style="red")
    
    def scan_existing_files(self):
        """Scan for existing files that haven't been processed"""
        console.print(f"🔍 Scanning for unprocessed files in: {self.data_dir}", style="blue")
        
        try:
            # Get all Excel and PDF files
            excel_files = list(self.data_dir.glob("*.xlsx")) + list(self.data_dir.glob("*.xls"))
            pdf_files = list(self.data_dir.glob("*.pdf"))
            all_files = excel_files + pdf_files
            
            if not all_files:
                console.print("📭 No Excel or PDF files found in data directory", style="yellow")
                return
            
            console.print(f"📋 Found {len(all_files)} files to check", style="cyan")
            
            # Check which files need processing
            processed_count = 0
            for file_path in all_files:
                filename = file_path.name
                
                # Skip QA files
                if "QA" in filename:
                    console.print(f"⏭️ Skipping QA file: {filename}", style="yellow")
                    continue
                
                # Check if already processed
                if filename in self.handler.processed_files:
                    console.print(f"✅ Already processed: {filename}", style="green")
                    continue
                
                # Process the file
                console.print(f"📤 Processing existing file: {filename}", style="blue")
                self.handler._process_new_file(str(file_path))
                processed_count += 1
            
            if processed_count == 0:
                console.print("✅ All files are already processed", style="green")
            else:
                console.print(f"✅ Processed {processed_count} new files", style="green")
                
        except Exception as e:
            console.print(f"❌ Error scanning existing files: {e}", style="red")

# Global file monitor instance
_file_monitor = None

def get_file_monitor() -> FileMonitor:
    """Get or create global file monitor instance"""
    global _file_monitor
    if _file_monitor is None:
        _file_monitor = FileMonitor()
    return _file_monitor
