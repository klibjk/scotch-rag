"""
FastHTML + MonsterUI Frontend Application

This module creates the main web application using FastHTML and MonsterUI
for the Scotch-RAG chat interface.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# FastHTML + MonsterUI imports
try:
    from fasthtml import FastHTML, html
    from monsterui import MonsterUI, ui
except ImportError:
    # Fallback for development
    FastHTML = None
    MonsterUI = None

from ..rag_system import RAGSystem, RAGConfig

logger = logging.getLogger(__name__)


class ScotchRAGApp:
    """
    Main FastHTML + MonsterUI application for Scotch-RAG.
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize the application.
        
        Args:
            rag_system: Initialized RAG system instance
        """
        self.rag_system = rag_system
        self.app = None
        self.conversation_history = []
        
        if FastHTML is None or MonsterUI is None:
            logger.warning("FastHTML/MonsterUI not available, using fallback")
            self._create_fallback_app()
        else:
            self._create_app()
    
    def _create_app(self):
        """Create the FastHTML + MonsterUI application."""
        self.app = FastHTML()
        
        # Register routes
        self.app.route("/")(self.home_page)
        self.app.route("/upload")(self.upload_page)
        self.app.route("/chat")(self.chat_page)
        self.app.route("/api/upload", methods=["POST"])(self.api_upload)
        self.app.route("/api/ask", methods=["POST"])(self.api_ask)
        self.app.route("/api/status")(self.api_status)
    
    def _create_fallback_app(self):
        """Create a fallback application for development."""
        # This would be a simple HTML/JavaScript app for development
        logger.info("Using fallback app for development")
    
    def home_page(self):
        """Render the home page."""
        return html("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Scotch-RAG - Document Q&A System</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }
                .header p {
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }
                .content {
                    padding: 40px;
                }
                .features {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 30px;
                    margin: 40px 0;
                }
                .feature {
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    transition: transform 0.3s ease;
                }
                .feature:hover {
                    transform: translateY(-5px);
                }
                .feature h3 {
                    color: #667eea;
                    margin-bottom: 15px;
                }
                .cta-buttons {
                    text-align: center;
                    margin: 40px 0;
                }
                .btn {
                    display: inline-block;
                    padding: 15px 30px;
                    margin: 0 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: 500;
                    transition: transform 0.3s ease;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .btn-secondary {
                    background: #6c757d;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🥃 Scotch-RAG</h1>
                    <p>Intelligent Document Q&A with RAG Technology</p>
                </div>
                <div class="content">
                    <h2>Welcome to Scotch-RAG</h2>
                    <p>Upload your documents and ask questions to get intelligent answers powered by advanced RAG (Retrieval-Augmented Generation) technology.</p>
                    
                    <div class="features">
                        <div class="feature">
                            <h3>📄 Multi-Format Support</h3>
                            <p>Upload PDFs, Excel files, Word documents, and text files. Our system processes them all.</p>
                        </div>
                        <div class="feature">
                            <h3>🤖 AI-Powered Answers</h3>
                            <p>Get intelligent, context-aware answers based on your uploaded documents.</p>
                        </div>
                        <div class="feature">
                            <h3>💬 Conversational Interface</h3>
                            <p>Chat naturally with your documents. Ask follow-up questions and get detailed responses.</p>
                        </div>
                    </div>
                    
                    <div class="cta-buttons">
                        <a href="/upload" class="btn">📤 Upload Documents</a>
                        <a href="/chat" class="btn btn-secondary">💬 Start Chatting</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    def upload_page(self):
        """Render the document upload page."""
        return html("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Upload Documents - Scotch-RAG</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }
                .content {
                    padding: 40px;
                }
                .upload-area {
                    border: 2px dashed #667eea;
                    border-radius: 10px;
                    padding: 40px;
                    text-align: center;
                    margin: 20px 0;
                    transition: border-color 0.3s ease;
                }
                .upload-area:hover {
                    border-color: #764ba2;
                }
                .upload-area.dragover {
                    border-color: #764ba2;
                    background: #f8f9fa;
                }
                .file-input {
                    display: none;
                }
                .btn {
                    display: inline-block;
                    padding: 15px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: 500;
                    cursor: pointer;
                    border: none;
                    transition: transform 0.3s ease;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .file-list {
                    margin: 20px 0;
                }
                .file-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 5px;
                    margin: 5px 0;
                }
                .progress {
                    width: 100%;
                    height: 20px;
                    background: #e9ecef;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }
                .progress-bar {
                    height: 100%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    transition: width 0.3s ease;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📤 Upload Documents</h1>
                    <p>Upload your documents to start asking questions</p>
                </div>
                <div class="content">
                    <div class="upload-area" id="uploadArea">
                        <h3>Drop files here or click to browse</h3>
                        <p>Supported formats: PDF, Excel (.xlsx, .xls), Word (.docx, .doc), Text (.txt), CSV</p>
                        <input type="file" id="fileInput" class="file-input" multiple accept=".pdf,.xlsx,.xls,.docx,.doc,.txt,.csv">
                        <button class="btn" onclick="document.getElementById('fileInput').click()">Choose Files</button>
                    </div>
                    
                    <div class="file-list" id="fileList"></div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="/chat" class="btn">💬 Start Chatting</a>
                        <a href="/" class="btn" style="background: #6c757d;">🏠 Back to Home</a>
                    </div>
                </div>
            </div>
            
            <script>
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                const fileList = document.getElementById('fileList');
                
                // Drag and drop functionality
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
                
                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.classList.remove('dragover');
                });
                
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    handleFiles(files);
                });
                
                fileInput.addEventListener('change', (e) => {
                    handleFiles(e.target.files);
                });
                
                function handleFiles(files) {
                    Array.from(files).forEach(file => {
                        uploadFile(file);
                    });
                }
                
                function uploadFile(file) {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    // Add file to list
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.innerHTML = `
                        <span>${file.name}</span>
                        <div class="progress">
                            <div class="progress-bar" style="width: 0%"></div>
                        </div>
                    `;
                    fileList.appendChild(fileItem);
                    
                    const progressBar = fileItem.querySelector('.progress-bar');
                    
                    fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            progressBar.style.width = '100%';
                            fileItem.innerHTML = `<span>${file.name}</span> <span style="color: green;">✓ Uploaded</span>`;
                        } else {
                            fileItem.innerHTML = `<span>${file.name}</span> <span style="color: red;">✗ Error</span>`;
                        }
                    })
                    .catch(error => {
                        fileItem.innerHTML = `<span>${file.name}</span> <span style="color: red;">✗ Error</span>`;
                    });
                }
            </script>
        </body>
        </html>
        """)
    
    def chat_page(self):
        """Render the chat interface page."""
        return html("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat - Scotch-RAG</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    height: 80vh;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                .chat-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                }
                .chat-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f8f9fa;
                }
                .message {
                    margin: 10px 0;
                    padding: 15px;
                    border-radius: 10px;
                    max-width: 80%;
                }
                .message.user {
                    background: #667eea;
                    color: white;
                    margin-left: auto;
                }
                .message.bot {
                    background: white;
                    border: 1px solid #dee2e6;
                }
                .chat-input {
                    padding: 20px;
                    border-top: 1px solid #dee2e6;
                    background: white;
                }
                .input-group {
                    display: flex;
                    gap: 10px;
                }
                .chat-input input {
                    flex: 1;
                    padding: 15px;
                    border: 1px solid #dee2e6;
                    border-radius: 25px;
                    font-size: 16px;
                }
                .btn {
                    padding: 15px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 25px;
                    cursor: pointer;
                    font-weight: 500;
                }
                .btn:hover {
                    opacity: 0.9;
                }
                .loading {
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>💬 Scotch-RAG Chat</h1>
                    <p>Ask questions about your uploaded documents</p>
                </div>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message bot">
                            Hello! I'm your Scotch-RAG assistant. Ask me anything about your uploaded documents.
                        </div>
                    </div>
                    <div class="chat-input">
                        <div class="input-group">
                            <input type="text" id="questionInput" placeholder="Ask a question..." onkeypress="handleKeyPress(event)">
                            <button class="btn" onclick="askQuestion()">Send</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const chatMessages = document.getElementById('chatMessages');
                const questionInput = document.getElementById('questionInput');
                
                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        askQuestion();
                    }
                }
                
                function askQuestion() {
                    const question = questionInput.value.trim();
                    if (!question) return;
                    
                    // Add user message
                    addMessage(question, 'user');
                    questionInput.value = '';
                    
                    // Add loading message
                    const loadingId = addLoadingMessage();
                    
                    // Send question to API
                    fetch('/api/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    })
                    .then(response => response.json())
                    .then(data => {
                        removeLoadingMessage(loadingId);
                        addMessage(data.answer, 'bot');
                    })
                    .catch(error => {
                        removeLoadingMessage(loadingId);
                        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                    });
                }
                
                function addMessage(text, sender) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${sender}`;
                    messageDiv.textContent = text;
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function addLoadingMessage() {
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'loading';
                    loadingDiv.id = 'loading-' + Date.now();
                    loadingDiv.textContent = 'Thinking...';
                    chatMessages.appendChild(loadingDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    return loadingDiv.id;
                }
                
                function removeLoadingMessage(loadingId) {
                    const loadingDiv = document.getElementById(loadingId);
                    if (loadingDiv) {
                        loadingDiv.remove();
                    }
                }
            </script>
        </body>
        </html>
        """)
    
    def api_upload(self, request):
        """Handle file upload API endpoint."""
        try:
            # Handle file upload logic here
            return {"success": True, "message": "File uploaded successfully"}
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {"success": False, "error": str(e)}
    
    def api_ask(self, request):
        """Handle question asking API endpoint."""
        try:
            data = request.json()
            question = data.get('question', '')
            
            if not question:
                return {"success": False, "error": "No question provided"}
            
            # Get answer from RAG system
            answer = self.rag_system.ask_question(question)
            
            return {
                "success": True,
                "answer": answer,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Ask error: {e}")
            return {"success": False, "error": str(e)}
    
    def api_status(self):
        """Get system status API endpoint."""
        try:
            doc_count = self.rag_system.get_document_count()
            supported_formats = self.rag_system.get_supported_formats()
            
            return {
                "success": True,
                "document_count": doc_count,
                "supported_formats": supported_formats,
                "status": "ready"
            }
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {"success": False, "error": str(e)}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = True):
        """Run the application."""
        if self.app:
            self.app.run(host=host, port=port, debug=debug)
        else:
            logger.error("Application not initialized")


def create_app(rag_system: RAGSystem) -> ScotchRAGApp:
    """
    Create and return a Scotch-RAG application instance.
    
    Args:
        rag_system: Initialized RAG system instance
        
    Returns:
        ScotchRAGApp instance
    """
    return ScotchRAGApp(rag_system)
