# Frontend Server Guide

## 🚀 Quick Start

### Method 1: Using the Main Application (Recommended)
```bash
python3 main_fasthtml.py
```

This is the **recommended way** to run the application. It:
- Automatically sets up both RAG systems (Pandas + LlamaParse)
- Loads environment variables
- Auto-ingests test documents
- Starts the web server

### Method 2: Direct Uvicorn (Advanced)
```bash
# First, set up environment
export OPENAI_API_KEY="your-api-key-here"

# Then run with uvicorn directly
uvicorn main_fasthtml:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Server Configuration

### Environment Variables
Create a `.env` file in the root directory:
```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional
LLAMA_CLOUD_API_KEY=your-llamacloud-api-key-here
HOST=0.0.0.0
PORT=8000
DEBUG=True

# RAG Configuration
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=50
RAG_MAX_RETRIEVAL=7
```

### Server Settings
- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `8000` (default)
- **Debug Mode**: `True` (auto-reload on file changes)
- **Framework**: FastHTML + Uvicorn (ASGI server)

## 🌐 Accessing the Application

Once the server is running, you can access:

- **Home Page**: http://localhost:8000/
- **Pandas Chat**: http://localhost:8000/chat-pandas
- **LlamaParse Chat**: http://localhost:8000/chat-llamaparse
- **Upload Page**: http://localhost:8000/upload

## 📡 API Endpoints

### Chat Endpoints
- `POST /api/ask-pandas` - Query using Pandas RAG system
- `POST /api/ask-llamaparse` - Query using LlamaParse RAG system
- `POST /api/ask` - Default query (uses Pandas)

### File Management
- `POST /api/upload` - Upload and ingest documents
- `GET /api/status` - Get system status

### Example API Usage
```bash
# Query the Pandas RAG system
curl -X POST http://localhost:8000/api/ask-pandas \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=How many products are there?"

# Query the LlamaParse RAG system
curl -X POST http://localhost:8000/api/ask-llamaparse \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What is the first product?"
```

## 🛠️ Development

### Running in Development Mode
```bash
# Enable debug mode (auto-reload)
export DEBUG=True
python3 main_fasthtml.py
```

### Running in Production Mode
```bash
# Disable debug mode
export DEBUG=False
python3 main_fasthtml.py
```

### Using Different Ports
```bash
# Run on port 8080
export PORT=8080
python3 main_fasthtml.py

# Or directly with uvicorn
uvicorn main_fasthtml:app --host 0.0.0.0 --port 8080
```

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Kill existing process on port 8000
sudo lsof -ti:8000 | xargs kill -9

# Or use a different port
export PORT=8080
python3 main_fasthtml.py
```

#### Missing Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install FastHTML specifically
pip install python-fasthtml
```

#### Environment Variables Not Loaded
```bash
# Check if .env file exists
ls -la .env

# Create .env file from template
cp config/config_env_example.txt .env
# Then edit .env with your API keys
```

#### RAG Systems Not Initializing
```bash
# Check logs for specific errors
tail -f scotch_rag.log

# Verify API keys are set
echo $OPENAI_API_KEY
```

### Log Files
- **Application Logs**: `scotch_rag.log` (in root directory)
- **Server Logs**: Displayed in terminal when running

## 📊 Server Architecture

```
main_fasthtml.py (Entry Point)
    ↓
Initialize RAG Systems
    ↓
Create Web Application (frontend/app.py)
    ↓
Start Uvicorn Server
    ↓
Serve FastHTML Application
```

### Components
- **FastHTML**: Web framework for Python
- **Uvicorn**: ASGI server for running FastHTML
- **MonsterUI**: UI components (optional)
- **RAG Systems**: Dual document processing engines

## 🔄 Auto-Ingestion

The application automatically ingests `rag/data/scotch_product_catalog.xlsx` on startup into both RAG systems. This ensures you have test data available immediately.

### Manual Ingestion
If you want to ingest different files:
1. Use the web interface: http://localhost:8000/upload
2. Or use the scripts: `python3 scripts/ingest_file.py`

## 🚀 Deployment

### Local Development
```bash
python3 main_fasthtml.py
```

### Production Deployment
```bash
# Set production environment
export DEBUG=False
export HOST=0.0.0.0
export PORT=8000

# Run with process manager (e.g., systemd, supervisor)
python3 main_fasthtml.py
```

### Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python3", "main_fasthtml.py"]
```

## 📝 Notes

- The server uses **FastHTML** as the web framework
- **Uvicorn** is the ASGI server that runs FastHTML
- Both RAG systems (Pandas + LlamaParse) are initialized on startup
- The application supports hot-reloading in debug mode
- All API endpoints return JSON responses
- File uploads are temporarily hardcoded to use test files

For more detailed information about the RAG systems, see the main `README.md`.
