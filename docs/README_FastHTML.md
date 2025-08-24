# 🥃 Scotch-RAG

**Intelligent Document Q&A with RAG Technology**

Scotch-RAG is a powerful document question-answering system that uses Retrieval-Augmented Generation (RAG) technology to provide intelligent, context-aware answers based on your uploaded documents.

## ✨ Features

- **📄 Multi-Format Support**: Upload PDFs, Excel files, Word documents, and text files
- **🤖 AI-Powered Answers**: Get intelligent, context-aware answers using OpenAI's GPT models
- **💬 Conversational Interface**: Chat naturally with your documents using a beautiful web interface
- **🔍 Advanced RAG**: Uses FAISS vector search and LangChain for optimal document retrieval
- **🚀 FastHTML + MonsterUI**: Pure Python, server-side rendered web interface
- **☁️ AWS Ready**: Designed for deployment on AWS infrastructure

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastHTML      │    │   RAG System    │    │   AWS Services  │
│   + MonsterUI   │◄──►│   (Backend)     │◄──►│   (Production)  │
│   (Frontend)    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/klibjk/scotch-rag.git
   cd scotch-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy the example environment file
   cp config/env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

4. **Set your OpenAI API key**
   ```bash
   # In .env file
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   python src/main.py
   ```

6. **Open your browser**
   ```
   http://localhost:8000
   ```

## 📁 Project Structure

```
scotch-rag/
├── src/
│   ├── rag_system/          # Core RAG system
│   │   ├── core.py         # Main RAG orchestrator
│   │   ├── document_processor.py  # Document processing
│   │   ├── vector_store.py # FAISS vector operations
│   │   └── query_engine.py # Question answering
│   ├── frontend/           # FastHTML + MonsterUI interface
│   │   └── app.py         # Web application
│   └── main.py            # Application entry point
├── data/
│   ├── uploads/           # User uploaded files
│   ├── processed/         # Processed documents
│   └── faiss_indexes/     # Vector store indexes
├── config/
│   ├── env.example        # Environment template
│   └── environment.local  # Local environment (gitignored)
├── tests/
│   └── test_rag_system.py # RAG system tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file or use `config/environment.local`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (Matching Original RAG System)
OPENAI_MODEL=gpt-5-mini
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2000
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=50

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### Supported Document Formats

- **PDF** (.pdf) - Using PyPDF2
- **Excel** (.xlsx, .xls) - Using openpyxl
- **Word** (.docx, .doc) - Using python-docx
- **Text** (.txt) - Plain text files
- **CSV** (.csv) - Comma-separated values

## 🧪 Testing

Run the RAG system test:

```bash
python tests/test_rag_system.py
```

This will:
1. Create a sample document
2. Process it through the RAG system
3. Test question answering
4. Verify all components work correctly

## 💻 Development

### Running in Development Mode

```bash
# Set debug mode
export DEBUG=True

# Run the application
python src/main.py
```

### Code Structure

- **RAG System**: Modular design with separate components for document processing, vector storage, and query answering
- **Frontend**: FastHTML + MonsterUI for pure Python web development
- **Configuration**: Environment-based configuration with validation
- **Testing**: Comprehensive test suite for all components

## 🚀 Deployment

### Local Development

```bash
python src/main.py
```

### AWS Deployment

The project is designed for AWS deployment with:

- **ECS/Fargate**: Containerized application deployment
- **S3**: Document storage and FAISS index storage
- **RDS**: Metadata storage (PostgreSQL)
- **CloudFront**: CDN for frontend assets
- **API Gateway**: REST API endpoints

See `rag-system-analysis/scotch-rag-project-plan.md` for detailed deployment instructions.

## 🔍 How It Works

1. **Document Upload**: Users upload documents through the web interface
2. **Document Processing**: Documents are parsed and text is extracted
3. **Text Chunking**: Text is split into manageable chunks
4. **Vector Embedding**: Chunks are converted to vector embeddings using OpenAI
5. **FAISS Storage**: Embeddings are stored in FAISS for fast similarity search
6. **Question Processing**: User questions are converted to embeddings
7. **Similarity Search**: Most relevant document chunks are retrieved
8. **Answer Generation**: GPT model generates answers based on retrieved context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Takedaxz/discord-stock-recommend-bot**: Original RAG implementation inspiration
- **FastHTML + MonsterUI**: Pure Python web framework by Jeremy Howard
- **LangChain**: RAG framework and utilities
- **OpenAI**: GPT models and embeddings
- **FAISS**: Vector similarity search

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the project plan in `rag-system-analysis/`

---

**Made with ❤️ for intelligent document processing**
