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
    from fasthtml.common import *
    from monsterui.core import FastHTML as MonsterUIFastHTML

    FastHTML_AVAILABLE = True
    MonsterUI_AVAILABLE = True
except ImportError:
    # Fallback for development
    FastHTML_AVAILABLE = False
    MonsterUI_AVAILABLE = False

from rag.rag_system import RAGSystem

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

        if not FastHTML_AVAILABLE or not MonsterUI_AVAILABLE:
            logger.warning("FastHTML/MonsterUI not available, using fallback")
            self._create_fallback_app()
        else:
            self._create_app()

    def _create_app(self):
        """Create the FastHTML + MonsterUI application."""
        self.app, self.rt = fast_app()

        # Register routes using the correct decorator pattern
        @self.rt("/")
        def home():
            return self.home_page()

        @self.rt("/upload")
        def upload():
            return self.upload_page()

        @self.rt("/chat")
        def chat():
            return self.chat_page()

        @self.rt("/api/upload")
        def api_upload(request):
            return self.api_upload(request)

        @self.rt("/api/ask")
        def api_ask(request):
            return self.api_ask(request)

        @self.rt("/api/status")
        def api_status(request):
            return self.api_status()

    def _create_fallback_app(self):
        """Create a fallback application for development."""
        # This would be a simple HTML/JavaScript app for development
        logger.info("Using fallback app for development")

    def home_page(self):
        """Render the home page."""
        return Titled(
            "Scotch-RAG - Document Q&A System",
            Div(
                Div(
                    H1("🥃 Scotch-RAG"),
                    P("Intelligent Document Q&A with RAG Technology"),
                    cls="header",
                ),
                Div(
                    H2("Welcome to Scotch-RAG"),
                    P(
                        "Upload your documents and ask questions to get intelligent answers powered by advanced RAG (Retrieval-Augmented Generation) technology."
                    ),
                    Div(
                        Div(
                            H3("📄 Multi-Format Support"),
                            P(
                                "Upload PDFs, Excel files, Word documents, and text files. Our system processes them all."
                            ),
                            cls="feature",
                        ),
                        Div(
                            H3("🤖 AI-Powered Answers"),
                            P(
                                "Get intelligent, context-aware answers based on your uploaded documents."
                            ),
                            cls="feature",
                        ),
                        Div(
                            H3("💬 Conversational Interface"),
                            P(
                                "Chat naturally with your documents. Ask follow-up questions and get detailed responses."
                            ),
                            cls="feature",
                        ),
                        cls="features",
                    ),
                    Div(
                        A("📤 Upload Documents", href="/upload", cls="btn"),
                        A("💬 Start Chatting", href="/chat", cls="btn btn-secondary"),
                        cls="cta-buttons",
                    ),
                    cls="content",
                ),
                cls="container",
            ),
            Style(
                """
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
            """
            ),
        )

    def upload_page(self):
        """Render the document upload page."""
        return Titled(
            "Upload Documents - Scotch-RAG",
            Div(
                Div(
                    H1("📤 Upload Documents"),
                    P("Upload your documents to start asking questions"),
                    cls="header",
                ),
                Div(
                    Div(
                        H3("Drop files here or click to browse"),
                        P(
                            "Supported formats: PDF, Excel (.xlsx, .xls), Word (.docx, .doc), Text (.txt), CSV"
                        ),
                        Button(
                            "Choose Files",
                            cls="btn",
                            onclick="document.getElementById('fileInput').click()",
                        ),
                        Input(
                            type="file",
                            id="fileInput",
                            cls="file-input",
                            multiple=True,
                            accept=".pdf,.xlsx,.xls,.docx,.doc,.txt,.csv",
                        ),
                        id="uploadArea",
                        cls="upload-area",
                    ),
                    Div(id="fileList", cls="file-list"),
                    Div(
                        A("💬 Start Chatting", href="/chat", cls="btn"),
                        A(
                            "🏠 Back to Home",
                            href="/",
                            cls="btn",
                            style="background: #6c757d;",
                        ),
                        style="text-align: center; margin-top: 30px;",
                    ),
                    cls="content",
                ),
                cls="container",
            ),
            Style(
                """
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
                    cursor: pointer;
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
                """
            ),
            Script(
                """
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
                """
            ),
        )

    def chat_page(self):
        """Render the chat interface page."""
        return Titled(
            "Chat - Scotch-RAG",
            Div(
                Div(
                    H1("💬 Scotch-RAG Chat"),
                    P("Ask questions about your uploaded documents"),
                    cls="header",
                ),
                Div(
                    Div(
                        Div(
                            "Hello! I'm your Scotch-RAG assistant. Ask me anything about your uploaded documents.",
                            cls="message bot",
                        ),
                        id="chatMessages",
                        cls="chat-messages",
                    ),
                    Div(
                        Div(
                            Input(
                                type="text",
                                id="questionInput",
                                placeholder="Ask a question...",
                                onkeypress="handleKeyPress(event)",
                            ),
                            Button("Send", cls="btn", onclick="askQuestion()"),
                            cls="input-group",
                        ),
                        cls="chat-input",
                    ),
                    cls="chat-container",
                ),
                cls="container",
            ),
            Style(
                """
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
                """
            ),
            Script(
                """
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
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: 'question=' + encodeURIComponent(question)
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
                """
            ),
        )

    def api_upload(self, request):
        """Handle file upload API endpoint."""
        try:
            # Get uploaded files from the request
            files = request.files()

            if not files:
                return JSONResponse({"success": False, "error": "No files uploaded"})

            uploaded_files = []
            for file in files:
                try:
                    # Save the uploaded file temporarily
                    import tempfile
                    import os

                    # Create uploads directory if it doesn't exist
                    upload_dir = Path("data/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    # Save file to uploads directory
                    file_path = upload_dir / file.filename
                    with open(file_path, "wb") as f:
                        f.write(file.file.read())

                    # Process the file with RAG system
                    result = self.rag_system.ingest_file(str(file_path))

                    if result.get("status") == "success":
                        uploaded_files.append(
                            {
                                "filename": file.filename,
                                "status": "success",
                                "chunks": result.get("chunks_created", 0),
                            }
                        )
                    else:
                        uploaded_files.append(
                            {
                                "filename": file.filename,
                                "status": "error",
                                "error": result.get("error", "Unknown error"),
                            }
                        )

                except Exception as e:
                    uploaded_files.append(
                        {"filename": file.filename, "status": "error", "error": str(e)}
                    )

            # Check if any files were successfully uploaded
            successful_uploads = [f for f in uploaded_files if f["status"] == "success"]

            if successful_uploads:
                return JSONResponse(
                    {
                        "success": True,
                        "message": f"Successfully uploaded {len(successful_uploads)} file(s)",
                        "files": uploaded_files,
                    }
                )
            else:
                return JSONResponse(
                    {
                        "success": False,
                        "error": "No files were successfully processed",
                        "files": uploaded_files,
                    }
                )

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    def api_ask(self, request):
        """Handle question asking API endpoint."""
        try:
            # Extract question from request
            question = ""

            # Try to get question from request body using asyncio
            if hasattr(request, "body"):
                try:
                    import asyncio

                    # Get the event loop and run the async body() method
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    body_data = loop.run_until_complete(request.body())
                    if isinstance(body_data, bytes):
                        body_data = body_data.decode("utf-8")

                    logger.info(f"Request body: {body_data}")

                    # Try to parse as form data first
                    if "question=" in body_data:
                        parts = body_data.split("question=")
                        if len(parts) > 1:
                            question = (
                                parts[1].split("&")[0] if "&" in parts[1] else parts[1]
                            )
                            question = question.replace("+", " ").replace("%20", " ")
                            logger.info(f"Parsed question from form data: {question}")

                    # If not form data, try JSON
                    elif not question and body_data.strip().startswith("{"):
                        import json

                        data = json.loads(body_data)
                        question = data.get("question", "")
                        logger.info(f"Parsed question from JSON: {question}")

                except Exception as e:
                    logger.error(f"Error parsing request body: {e}")

            # If still no question, try other methods
            if not question:
                try:
                    # Try to get from query parameters
                    if hasattr(request, "query_params"):
                        question = request.query_params.get("question", "")
                        logger.info(f"Got question from query params: {question}")

                    # Try to get from request parameters
                    elif hasattr(request, "params"):
                        question = request.params.get("question", "")
                        logger.info(f"Got question from params: {question}")

                except Exception as e:
                    logger.error(f"Error getting question from params: {e}")

            if not question:
                return JSONResponse({"success": False, "error": "No question provided"})

            logger.info(f"Processing question: {question}")

            # Get answer from RAG system
            result = self.rag_system.query(question)

            logger.info(f"RAG result status: {result.get('status')}")

            if result["status"] == "success":
                answer = result["answer"]
                logger.info(f"Generated answer length: {len(answer)}")
                return JSONResponse(
                    {"success": True, "answer": answer, "question": question}
                )
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"RAG query failed: {error_msg}")
                return JSONResponse({"success": False, "error": error_msg})

        except Exception as e:
            logger.error(f"Ask error: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    def api_status(self):
        """Get system status API endpoint."""
        try:
            stats = self.rag_system.get_stats()

            return JSONResponse(
                {
                    "success": True,
                    "document_count": stats.get("total_vectors", 0),
                    "supported_formats": [
                        "PDF",
                        "Excel (.xlsx, .xls)",
                        "Word (.docx, .doc)",
                        "Text (.txt)",
                        "CSV",
                    ],
                    "status": (
                        "ready" if stats.get("status") == "active" else "no_documents"
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Status error: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = True):
        """Run the application."""
        if self.app:
            import uvicorn

            uvicorn.run(
                self.app, host=host, port=port, log_level="debug" if debug else "info"
            )
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
