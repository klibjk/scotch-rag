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
from rag.rag_system_llamaparse import RAGSystemLlamaParse

logger = logging.getLogger(__name__)


class ScotchRAGApp:
    """
    Main FastHTML + MonsterUI application for Scotch-RAG.
    """

    def __init__(
        self, rag_system_pandas: RAGSystem, rag_system_llamaparse: RAGSystemLlamaParse
    ):
        """
        Initialize the application.

        Args:
            rag_system_pandas: Initialized pandas-based RAG system instance
            rag_system_llamaparse: Initialized LlamaParse-based RAG system instance
        """
        self.rag_system_pandas = rag_system_pandas
        self.rag_system_llamaparse = rag_system_llamaparse
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

        @self.rt("/chat-pandas")
        def chat_pandas():
            return self.chat_page("pandas")

        @self.rt("/chat-llamaparse")
        def chat_llamaparse():
            return self.chat_page("llamaparse")

        @self.rt("/api/upload")
        def api_upload(request):
            return self.api_upload(request)

        @self.rt("/api/ask")
        def api_ask(request):
            return self.api_ask(request, "pandas")

        @self.rt("/api/ask-pandas")
        def api_ask_pandas(request):
            return self.api_ask(request, "pandas")

        @self.rt("/api/ask-llamaparse")
        def api_ask_llamaparse(request):
            return self.api_ask(request, "llamaparse")

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
                        H3(
                            "📚 Test Document Ready",
                            style="text-align: center; margin-bottom: 20px; color: #333;",
                        ),
                        P(
                            "The test document (scotch_product_catalog.xlsx) is already loaded into both RAG systems. Choose your preferred parser to start chatting:",
                            style="text-align: center; color: #666; margin-bottom: 30px;",
                        ),
                        H3(
                            "Choose Your Parser:",
                            style="text-align: center; margin-bottom: 20px; color: #333;",
                        ),
                        Div(
                            Div(
                                H4(
                                    "🐼 Pandas Parser",
                                    style="color: #667eea; margin-bottom: 10px;",
                                ),
                                P(
                                    "Fast Excel processing with simple text conversion",
                                    style="color: #666; margin-bottom: 15px;",
                                ),
                                A(
                                    "🚀 Start Pandas Chat",
                                    href="/chat-pandas",
                                    cls="btn",
                                    style="background: #667eea;",
                                ),
                                cls="parser-option",
                            ),
                            Div(
                                H4(
                                    "🦙 LlamaParse Parser",
                                    style="color: #f5576c; margin-bottom: 10px;",
                                ),
                                P(
                                    "Enhanced document understanding with structured data extraction",
                                    style="color: #666; margin-bottom: 15px;",
                                ),
                                A(
                                    "🚀 Start LlamaParse Chat",
                                    href="/chat-llamaparse",
                                    cls="btn",
                                    style="background: #f5576c;",
                                ),
                                cls="parser-option",
                            ),
                            cls="parser-options",
                        ),
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
                .parser-options {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin-top: 20px;
                }
                .parser-option {
                    background: #f8f9fa;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    border: 2px solid transparent;
                    transition: all 0.3s ease;
                }
                .parser-option:hover {
                    border-color: #667eea;
                    transform: translateY(-2px);
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
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

    def chat_page(self, parser_type="pandas"):
        """Render the chat interface page."""
        parser_name = "Pandas" if parser_type == "pandas" else "LlamaParse"
        api_endpoint = (
            f"/api/ask-{parser_type}" if parser_type != "pandas" else "/api/ask"
        )
        return Titled(
            "Chat - Scotch-RAG",
            Div(
                Div(
                    H1(f"💬 Scotch-RAG Chat ({parser_name})"),
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
                    Button(
                        "↑",
                        id="scrollToTopBtn",
                        cls="scroll-to-top-btn",
                        onclick="scrollToTop()",
                        title="Scroll to top",
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
                    padding: 5px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    max-width: 95vw;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    height: 95vh;
                    min-height: 700px;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 10px;
                    text-align: center;
                    flex-shrink: 0;
                }
                .chat-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    position: relative;
                }
                .chat-messages {
                    flex: 1;
                    padding: 10px;
                    overflow-y: scroll;
                    background: #f8f9fa;
                    scroll-behavior: smooth;
                    scrollbar-width: thin;
                    scrollbar-color: #667eea #f8f9fa;
                    min-height: 400px;
                    max-height: 70vh;
                }
                .chat-messages::-webkit-scrollbar {
                    width: 12px;
                    background-color: #f8f9fa;
                }
                .chat-messages::-webkit-scrollbar-track {
                    background: #f8f9fa;
                    border-radius: 6px;
                }
                .chat-messages::-webkit-scrollbar-thumb {
                    background: #667eea;
                    border-radius: 6px;
                    border: 2px solid #f8f9fa;
                }
                .chat-messages::-webkit-scrollbar-thumb:hover {
                    background: #764ba2;
                }
                .chat-messages::-webkit-scrollbar-corner {
                    background: #f8f9fa;
                }
                .message {
                    margin: 8px 0;
                    padding: 10px 12px;
                    border-radius: 10px;
                    max-width: 85%;
                    word-wrap: break-word;
                    line-height: 1.4;
                    position: relative;
                    animation: fadeIn 0.3s ease-in;
                }
                .message.user {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    margin-left: auto;
                    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
                }
                .message.bot {
                    background: white;
                    border: 1px solid #dee2e6;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }
                .message-timestamp {
                    font-size: 10px;
                    opacity: 0.6;
                    margin-top: 3px;
                    display: block;
                }
                .message.user .message-timestamp {
                    color: rgba(255, 255, 255, 0.8);
                }
                .message.bot .message-timestamp {
                    color: #6c757d;
                }
                .chat-input {
                    padding: 12px;
                    border-top: 1px solid #dee2e6;
                    background: white;
                    flex-shrink: 0;
                }
                .input-group {
                    display: flex;
                    gap: 10px;
                }
                .chat-input input {
                    flex: 1;
                    padding: 10px 15px;
                    border: 1px solid #dee2e6;
                    border-radius: 20px;
                    font-size: 14px;
                    transition: border-color 0.3s ease;
                }
                .chat-input input:focus {
                    outline: none;
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }
                .btn {
                    padding: 10px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 20px;
                    cursor: pointer;
                    font-weight: 500;
                    font-size: 14px;
                    transition: all 0.3s ease;
                }
                .btn:hover {
                    opacity: 0.9;
                    transform: translateY(-1px);
                }
                .btn:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                    transform: none;
                }
                .loading {
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                }
                .loading-spinner {
                    width: 20px;
                    height: 20px;
                    border: 2px solid #f3f3f3;
                    border-top: 2px solid #667eea;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                .scroll-to-top-btn {
                    position: absolute;
                    bottom: 80px;
                    right: 15px;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: bold;
                    box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
                    transition: all 0.3s ease;
                    opacity: 0;
                    visibility: hidden;
                    z-index: 1000;
                }
                .scroll-to-top-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                }
                .scroll-to-top-btn.visible {
                    opacity: 1;
                    visibility: visible;
                }
                .error-message {
                    background: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .retry-btn {
                    background: #dc3545;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 12px;
                }
                .retry-btn:hover {
                    background: #c82333;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                """
            ),
            Script(
                """
                const chatMessages = document.getElementById('chatMessages');
                const questionInput = document.getElementById('questionInput');
                const scrollToTopBtn = document.getElementById('scrollToTopBtn');
                const sendBtn = document.querySelector('.btn');
                
                // Initialize scroll functionality
                chatMessages.addEventListener('scroll', handleScroll);
                
                function handleKeyPress(event) {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        askQuestion();
                    }
                }
                
                function askQuestion() {
                    const question = questionInput.value.trim();
                    if (!question) return;
                    
                    // Disable input and button during request
                    questionInput.disabled = true;
                    sendBtn.disabled = true;
                    
                    // Add user message
                    addMessage(question, 'user');
                    questionInput.value = '';
                    
                    // Add loading message
                    const loadingId = addLoadingMessage();
                    
                    // Send question to API
                    fetch('"""
                + api_endpoint
                + """', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: 'question=' + encodeURIComponent(question)
                    })
                    .then(response => response.json())
                    .then(data => {
                        removeLoadingMessage(loadingId);
                        if (data.success) {
                            addMessage(data.answer, 'bot');
                        } else {
                            addErrorMessage(data.error || 'An error occurred', question);
                        }
                    })
                    .catch(error => {
                        removeLoadingMessage(loadingId);
                        addErrorMessage('Network error. Please check your connection.', question);
                    })
                    .finally(() => {
                        // Re-enable input and button
                        questionInput.disabled = false;
                        sendBtn.disabled = false;
                        questionInput.focus();
                    });
                }
                
                function addMessage(text, sender) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${sender}`;
                    
                    // Create message content with timestamp
                    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    messageDiv.innerHTML = `
                        <div>${escapeHtml(text)}</div>
                        <span class="message-timestamp">${timestamp}</span>
                    `;
                    
                    chatMessages.appendChild(messageDiv);
                    scrollToBottom();
                }
                
                function addErrorMessage(error, originalQuestion) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error-message';
                    errorDiv.innerHTML = `
                        <span>⚠️ ${escapeHtml(error)}</span>
                        <button class="retry-btn" onclick="retryQuestion('${escapeHtml(originalQuestion)}')">Retry</button>
                    `;
                    chatMessages.appendChild(errorDiv);
                    scrollToBottom();
                }
                
                function retryQuestion(question) {
                    // Remove the error message
                    const errorMessage = document.querySelector('.error-message');
                    if (errorMessage) {
                        errorMessage.remove();
                    }
                    
                    // Set the question back in the input and retry
                    questionInput.value = question;
                    askQuestion();
                }
                
                function addLoadingMessage() {
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'loading';
                    loadingDiv.id = 'loading-' + Date.now();
                    loadingDiv.innerHTML = `
                        <div class="loading-spinner"></div>
                        <span>Thinking...</span>
                    `;
                    chatMessages.appendChild(loadingDiv);
                    scrollToBottom();
                    return loadingDiv.id;
                }
                
                function removeLoadingMessage(loadingId) {
                    const loadingDiv = document.getElementById(loadingId);
                    if (loadingDiv) {
                        loadingDiv.remove();
                    }
                }
                
                function scrollToBottom() {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function scrollToTop() {
                    chatMessages.scrollTo({
                        top: 0,
                        behavior: 'smooth'
                    });
                }
                
                function handleScroll() {
                    // Show/hide scroll to top button based on scroll position
                    if (chatMessages.scrollTop > 300) {
                        scrollToTopBtn.classList.add('visible');
                    } else {
                        scrollToTopBtn.classList.remove('visible');
                    }
                }
                
                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }
                
                // Focus input on page load
                window.addEventListener('load', () => {
                    questionInput.focus();
                });
                """
            ),
        )

    def api_upload(self, request):
        """Handle file upload API endpoint."""
        try:
            # For now, let's create a simple test to see what we're getting
            try:
                body_data = request.body()
                logger.info(f"Request body type: {type(body_data)}")
                logger.info(f"Request body: {body_data}")
            except Exception as e:
                logger.info(f"Could not get request body: {e}")

            # Since FastHTML doesn't handle multipart form data like FastAPI,
            # let's create a simple test upload that works with the existing file
            # For now, we'll use the existing scotch_product_catalog.xlsx file

            test_file_path = "rag/data/scotch_product_catalog.xlsx"

            if not os.path.exists(test_file_path):
                return JSONResponse({"success": False, "error": "Test file not found"})

            # Process the test file with both RAG systems
            pandas_result = self.rag_system_pandas.ingest_file(test_file_path)
            llamaparse_result = self.rag_system_llamaparse.ingest_file(test_file_path)

            # Check if both systems processed successfully
            pandas_success = pandas_result.get("status") == "success"
            llamaparse_success = llamaparse_result.get("status") == "success"

            if pandas_success and llamaparse_success:
                return JSONResponse(
                    {
                        "success": True,
                        "message": "Successfully ingested test file into both Pandas and LlamaParse systems",
                        "files": [
                            {
                                "filename": "scotch_product_catalog.xlsx",
                                "status": "success",
                                "pandas_chunks": pandas_result.get("chunks_created", 0),
                                "llamaparse_chunks": llamaparse_result.get(
                                    "chunks_created", 0
                                ),
                                "message": "Successfully ingested into both systems",
                            }
                        ],
                    }
                )
            elif pandas_success:
                return JSONResponse(
                    {
                        "success": True,
                        "message": "Successfully ingested test file into Pandas system",
                        "files": [
                            {
                                "filename": "scotch_product_catalog.xlsx",
                                "status": "partial_success",
                                "pandas_chunks": pandas_result.get("chunks_created", 0),
                                "llamaparse_error": llamaparse_result.get(
                                    "error", "Unknown error"
                                ),
                                "message": "Successfully ingested into Pandas system, LlamaParse failed",
                            }
                        ],
                    }
                )
            else:
                return JSONResponse(
                    {
                        "success": False,
                        "error": "Failed to ingest test file",
                        "files": [
                            {
                                "filename": "scotch_product_catalog.xlsx",
                                "status": "error",
                                "pandas_error": pandas_result.get(
                                    "error", "Unknown error"
                                ),
                                "llamaparse_error": llamaparse_result.get(
                                    "error", "Unknown error"
                                ),
                                "message": "Failed to ingest into both systems",
                            }
                        ],
                    }
                )

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    def api_ask(self, request, parser_type="pandas"):
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

            # Get answer from appropriate RAG system
            if parser_type == "llamaparse":
                result = self.rag_system_llamaparse.query(question)
            else:
                result = self.rag_system_pandas.query(question)

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
            # Get stats from both systems
            stats_pandas = self.rag_system_pandas.get_stats()
            stats_llamaparse = self.rag_system_llamaparse.get_stats()

            # Use pandas stats for now (could be enhanced to show both)
            stats = stats_pandas

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


def create_app(
    rag_system_pandas: RAGSystem, rag_system_llamaparse: RAGSystemLlamaParse
) -> ScotchRAGApp:
    """
    Create and return a Scotch-RAG application instance.

    Args:
        rag_system_pandas: Initialized pandas-based RAG system instance
        rag_system_llamaparse: Initialized LlamaParse-based RAG system instance

    Returns:
        ScotchRAGApp instance
    """
    return ScotchRAGApp(rag_system_pandas, rag_system_llamaparse)
