"""
Enhanced RAG System with LlamaParse (Alternative to pandas-based system)
=======================================================================
Handles Excel file ingestion using LlamaParse for enhanced document parsing.
Uses the same framework as the reference project but with LlamaParse as primary parser.
Enhanced with query optimization, validation, and quality improvements.
"""

import os
import json
import asyncio
import pickle
import re
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import shutil
from dataclasses import dataclass

# Import LlamaParse (primary parser)
try:
    from llama_parse import LlamaParse

    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False
    print("⚠️ LlamaParse not available. Install with: pip install llama-parse")

# Import Sentence Transformers with fallback
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "⚠️ Sentence Transformers not available. Install with: pip install sentence-transformers"
    )

# Import Anthropic with fallback
try:
    from langchain_anthropic import ChatAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ Anthropic not available. Install with: pip install langchain-anthropic")

# Import Google with fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print(
        "⚠️ Google GenAI not available. Install with: pip install langchain-google-genai"
    )

# Import Pinecone with fallback
try:
    from langchain_pinecone import PineconeVectorStore

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not available. Install with: pip install langchain-pinecone")

# Import Redis with fallback
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available. Install with: pip install redis")

# Import Unstructured with fallback
try:
    from langchain_community.document_loaders import UnstructuredFileLoader

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ Unstructured not available. Install with: pip install unstructured")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv


@dataclass
class QueryAnalysis:
    """Query analysis result for quality enhancement"""

    original_query: str
    enhanced_query: str
    keywords: List[str]
    intent: str
    language: str
    confidence: float
    suggestions: List[str]


@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics"""

    query_time: float
    retrieval_time: float
    generation_time: float
    total_tokens: int
    source_count: int
    relevance_score: float
    confidence_score: float


# Optional web search
try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

load_dotenv()


class RAGSystemLlamaParse:
    def __init__(
        self,
        base_storage_dir: str = "./faiss_indexes_llamaparse",
        chunk_size: int = 800,
        chunk_overlap: int = 50,
        max_retrieval_results: int = 7,
    ):

        self.base_storage_dir = Path(base_storage_dir)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI components immediately with performance optimizations
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            show_progress_bar=False,
            max_retries=2,  # Reduced retries for faster failure
            request_timeout=30,  # Add timeout
        )
        print("🔗 Using OpenAI embeddings (text-embedding-3-small)")

        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.1,
            max_retries=2,  # Reduced retries
            max_tokens=1000,  # Reduced tokens for faster response
            request_timeout=30,  # Add timeout
        )
        print(f"✅ Primary LLM initialized: {self.llm.model_name}")

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,  # Smaller chunks for more precise retrieval
            chunk_overlap=100,  # Increased overlap for better context
            length_function=len,
        )

        self.max_retrieval_results = 5  # Reduced from 7 for faster retrieval

        # Initialize Tavily if available
        self.tavily_client = None
        if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
            try:
                self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            except Exception as e:
                print(f"Warning: Could not initialize Tavily: {e}")

        # Cache for loaded indexes to avoid repeated file I/O
        self._index_cache = {}
        self._vectorstore_cache = None  # Cache the vectorstore
        self._qa_chain_cache = {}  # Cache QA chains
        self._initialized = False

        # Initialize LlamaParse for enhanced document parsing
        self.llama_parser = None
        if LLAMA_PARSE_AVAILABLE:
            llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
            if llama_api_key:
                try:
                    self.llama_parser = LlamaParse(
                        api_key=llama_api_key,
                        result_type="markdown",
                        verbose=True,
                        num_workers=4,
                        check_interval=1,
                    )
                    print("✅ LlamaParse initialized for enhanced document parsing!")
                except Exception as e:
                    print(f"⚠️ LlamaParse initialization failed: {e}")
                    print("📄 Will fall back to standard loaders")
            else:
                print(
                    "⚠️ LLAMA_CLOUD_API_KEY not found. Set it to enable LlamaParse for Excel files."
                )
        else:
            print("⚠️ LlamaParse package not installed. Run: pip install llama-parse")

        # Quality enhancement components
        self.query_cache = {}
        self.performance_metrics = []
        self.quality_thresholds = {
            "min_confidence": 0.3,
            "min_relevance": 0.5,
            "max_response_time": 10.0,
        }

        # Initialize fallback components immediately
        self.llm_fallbacks = []
        self.embedding_fallbacks = []
        self.vectorstore_fallbacks = []
        self.cache_fallbacks = []
        self.current_llm_index = 0
        self.current_embedding_index = 0
        self.current_vectorstore_index = 0
        self.current_cache_index = 0

        # Initialize fallbacks
        self._initialize_llm_fallbacks()
        self._initialize_embedding_fallbacks()
        self._initialize_vectorstore_fallbacks()
        self._initialize_cache_fallbacks()

        print("✅ LlamaParse RAG system components initialized!")
        print(f"🔄 LLM Fallbacks: {len(self.llm_fallbacks)} available")
        print(f"🔄 Embedding Fallbacks: {len(self.embedding_fallbacks)} available")
        print(f"🔄 Vectorstore Fallbacks: {len(self.vectorstore_fallbacks)} available")
        print(f"🔄 Cache Fallbacks: {len(self.cache_fallbacks)} available")

    def _initialize_llm_fallbacks(self):
        """Initialize LLM fallbacks"""
        try:
            # Fallback 1: Anthropic Claude Sonnet-4
            if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
                try:
                    anthropic_llm = ChatAnthropic(
                        model="claude-sonnet-4-20250514",
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    self.llm_fallbacks.append(
                        {
                            "name": "Anthropic Claude Sonnet-4",
                            "llm": anthropic_llm,
                            "priority": 1,
                        }
                    )
                    print("✅ Anthropic Claude Sonnet-4 fallback initialized")
                except Exception as e:
                    print(f"⚠️ Anthropic fallback failed: {e}")

            # Fallback 2: Google Gemini Pro
            if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
                try:
                    google_llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        temperature=0.1,
                        max_output_tokens=2000,
                    )
                    self.llm_fallbacks.append(
                        {"name": "Google Gemini Pro", "llm": google_llm, "priority": 2}
                    )
                    print("✅ Google Gemini Pro fallback initialized")
                except Exception as e:
                    print(f"⚠️ Google fallback failed: {e}")

        except Exception as e:
            print(f"⚠️ LLM fallback initialization failed: {e}")

    def _initialize_embedding_fallbacks(self):
        """Initialize embedding fallbacks"""
        try:
            # Fallback 1: Sentence Transformers
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    sentence_embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={"device": "cpu"},
                    )
                    self.embedding_fallbacks.append(
                        {
                            "name": "Sentence Transformers (all-MiniLM-L6-v2)",
                            "embeddings": sentence_embeddings,
                            "priority": 1,
                        }
                    )
                    print("✅ Sentence Transformers fallback initialized")
                except Exception as e:
                    print(f"⚠️ Sentence Transformers fallback failed: {e}")

        except Exception as e:
            print(f"⚠️ Embedding fallback initialization failed: {e}")

    def _initialize_vectorstore_fallbacks(self):
        """Initialize vectorstore fallbacks"""
        try:
            # Fallback 1: Pinecone (if API key available)
            if PINECONE_AVAILABLE and os.getenv("PINECONE_API_KEY"):
                try:
                    # Note: Pinecone requires index name and environment
                    # This is a placeholder - actual implementation would need index setup
                    self.vectorstore_fallbacks.append(
                        {"name": "Pinecone", "type": "pinecone", "priority": 1}
                    )
                    print("✅ Pinecone fallback configured (requires index setup)")
                except Exception as e:
                    print(f"⚠️ Pinecone fallback failed: {e}")

        except Exception as e:
            print(f"⚠️ Vectorstore fallback initialization failed: {e}")

    def _initialize_cache_fallbacks(self):
        """Initialize cache fallbacks"""
        try:
            # Fallback 1: Redis cache
            if REDIS_AVAILABLE:
                try:
                    redis_client = redis.Redis(
                        host=os.getenv("REDIS_HOST", "localhost"),
                        port=int(os.getenv("REDIS_PORT", 6379)),
                        db=0,
                        decode_responses=True,
                    )
                    # Test connection
                    redis_client.ping()
                    self.cache_fallbacks.append(
                        {"name": "Redis", "client": redis_client, "priority": 1}
                    )
                    print("✅ Redis cache fallback initialized")
                except Exception as e:
                    print(f"⚠️ Redis fallback failed: {e}")

        except Exception as e:
            print(f"⚠️ Cache fallback initialization failed: {e}")

    def get_fallback_llm(self):
        """Get next available LLM fallback"""
        if not self.llm_fallbacks:
            return None

        # Try current fallback
        if self.current_llm_index < len(self.llm_fallbacks):
            fallback = self.llm_fallbacks[self.current_llm_index]
            print(f"🔄 Using LLM fallback: {fallback['name']}")
            return fallback["llm"]

        # Reset to first fallback
        self.current_llm_index = 0
        if self.llm_fallbacks:
            fallback = self.llm_fallbacks[0]
            print(f"🔄 Using LLM fallback: {fallback['name']}")
            return fallback["llm"]

        return None

    def get_fallback_embeddings(self):
        """Get next available embedding fallback"""
        if not self.embedding_fallbacks:
            return None

        # Try current fallback
        if self.current_embedding_index < len(self.embedding_fallbacks):
            fallback = self.embedding_fallbacks[self.current_embedding_index]
            print(f"🔄 Using embedding fallback: {fallback['name']}")
            return fallback["embeddings"]

        # Reset to first fallback
        self.current_embedding_index = 0
        if self.embedding_fallbacks:
            fallback = self.embedding_fallbacks[0]
            print(f"🔄 Using embedding fallback: {fallback['name']}")
            return fallback["embeddings"]

        return None

    def get_fallback_cache(self, key: str):
        """Get value from fallback cache"""
        if not self.cache_fallbacks:
            return None

        for fallback in self.cache_fallbacks:
            try:
                if fallback["name"] == "Redis":
                    value = fallback["client"].get(key)
                    if value:
                        return json.loads(value)
            except Exception as e:
                print(f"⚠️ Cache fallback error: {e}")
                continue

        return None

    def set_fallback_cache(self, key: str, value: any, ttl: int = 3600):
        """Set value in fallback cache"""
        if not self.cache_fallbacks:
            return False

        for fallback in self.cache_fallbacks:
            try:
                if fallback["name"] == "Redis":
                    fallback["client"].setex(key, ttl, json.dumps(value))
                    return True
            except Exception as e:
                print(f"⚠️ Cache fallback error: {e}")
                continue

        return False

    def ingest_file(self, file_path: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Ingest a file into the RAG system using LlamaParse

        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata to add to documents

        Returns:
            Dict with ingestion results
        """
        try:
            print(f"📄 Processing file with LlamaParse: {file_path}")

            # Determine file type
            file_extension = Path(file_path).suffix.lower()

            if file_extension in [".xlsx", ".xls"]:
                documents = self._process_excel_file_llamaparse(file_path)
            elif file_extension == ".pdf":
                documents = self._process_pdf_file_llamaparse(file_path)
            else:
                raise Exception(f"Unsupported file type: {file_extension}")

            if not documents:
                raise Exception(f"No content found in {file_extension} file")

            # Prepare metadata
            doc_metadata = {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_type": file_extension,
                "parser": "llamaparse",
                **(metadata or {}),
            }

            # Update metadata for all documents
            for doc in documents:
                doc.metadata.update(doc_metadata)

            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            print(f"✂️ Created {len(texts)} chunks with LlamaParse")

            # Load existing index or create new one
            existing_vectorstore = self.load_index()

            if existing_vectorstore is None:
                # Create new vectorstore
                print(f"🆕 Creating new vectorstore with LlamaParse data")
                vectorstore = FAISS.from_documents(texts, self.embeddings)
            else:
                # Add to existing vectorstore
                print(f"📚 Adding LlamaParse data to existing vectorstore")
                vectorstore = existing_vectorstore
                vectorstore.add_documents(texts)

            # Save the updated vectorstore
            self.save_index(vectorstore)

            # Clear caches since we have new data
            self._vectorstore_cache = None
            self._qa_chain_cache = {}

            # Calculate total characters
            total_chars = sum(len(doc.page_content) for doc in documents)

            result = {
                "status": "success",
                "filename": doc_metadata["filename"],
                "chunks_created": len(texts),
                "total_characters": total_chars,
                "pages_processed": len(documents),
                "parser": "llamaparse",
                "metadata": doc_metadata,
            }

            print(
                f"✅ Successfully ingested {doc_metadata['filename']} with LlamaParse"
            )
            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "filename": os.path.basename(file_path) if file_path else "unknown",
                "parser": "llamaparse",
            }
            print(f"❌ LlamaParse ingestion failed: {e}")
            return error_result

    def _process_excel_file_llamaparse(self, file_path: str) -> List[Document]:
        """Process Excel file using LlamaParse as primary parser with pandas fallback"""
        print(f"📊 Processing Excel file with LlamaParse: {file_path}")

        # Use LlamaParse as primary parser
        if self.llama_parser:
            try:
                print(f"📊 Using LlamaParse for enhanced Excel processing: {file_path}")

                # Try LlamaParse (with minimal retry only if empty)
                llama_docs = self.llama_parser.load_data(file_path)

                if not llama_docs:
                    print(f"⚠️ LlamaParse returned empty result, trying once more...")
                    llama_docs = self.llama_parser.load_data(file_path)

                print(
                    f"📊 LlamaParse successfully loaded {len(llama_docs) if llama_docs else 0} structured chunks from Excel"
                )

                # Debug: Check what we got from LlamaParse
                if llama_docs:
                    print(f"📊 First document attributes: {dir(llama_docs[0])}")
                    if hasattr(llama_docs[0], "text"):
                        print(
                            f"📊 First document text length: {len(llama_docs[0].text)}"
                        )
                        print(
                            f"📊 First document text preview: {llama_docs[0].text[:200]}..."
                        )

                # Convert LlamaParse documents to LangChain Document format
                documents = []
                for i, doc in enumerate(llama_docs):
                    # LlamaParse documents have 'text' attribute instead of 'page_content'
                    content = doc.text if hasattr(doc, "text") else str(doc)

                    # Skip empty documents
                    if not content or not content.strip():
                        print(f"⚠️ Skipping empty document {i}")
                        continue

                    # Start with LlamaParse metadata if available
                    doc_metadata = (
                        doc.metadata.copy()
                        if hasattr(doc, "metadata") and doc.metadata
                        else {}
                    )

                    # Add our own metadata
                    doc_metadata.update(
                        {"source_index": i, "sheet": i, "parser": "llamaparse"}
                    )

                    documents.append(
                        Document(page_content=content.strip(), metadata=doc_metadata)
                    )

                print(
                    f"📊 Converted {len(documents)} non-empty Excel documents to LangChain format"
                )

                # If no documents after filtering, try a different approach
                if not documents and llama_docs:
                    print(
                        "🔄 No content found after conversion, trying alternative extraction..."
                    )
                    # Try to extract any available content
                    for i, doc in enumerate(llama_docs):
                        # Try different possible attributes
                        content = ""
                        if hasattr(doc, "text") and doc.text:
                            content = doc.text
                        elif hasattr(doc, "content") and doc.content:
                            content = doc.content
                        elif hasattr(doc, "page_content") and doc.page_content:
                            content = doc.page_content
                        else:
                            # Last resort - convert to string
                            content = str(doc)

                        if content and content.strip():
                            metadata = {
                                "sheet": i,
                                "extraction_method": "alternative",
                                "parser": "llamaparse",
                            }
                            documents.append(
                                Document(
                                    page_content=content.strip(), metadata=metadata
                                )
                            )

                    print(
                        f"📊 Alternative extraction yielded {len(documents)} documents"
                    )

                if documents:
                    return documents

            except Exception as llama_error:
                print(f"⚠️ LlamaParse failed: {llama_error}")
                print("🔄 Falling back to pandas processing...")

        # Fallback to pandas processing if LlamaParse fails
        print("📊 Using pandas fallback for Excel processing")
        try:
            import pandas as pd

            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            print(f"📋 Excel sheets found: {excel_file.sheet_names}")

            documents = []
            for sheet_name in excel_file.sheet_names:
                print(f"📋 Processing sheet: {sheet_name}")

                # Read sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(
                    f"📋 Sheet {sheet_name} has {len(df)} rows and {len(df.columns)} columns"
                )

                # Convert DataFrame to text
                text_content = f"Sheet: {sheet_name}\n"
                text_content += f"Columns: {', '.join(df.columns)}\n"
                text_content += f"Total rows: {len(df)}\n\n"

                # Add all data rows
                for idx, row in df.iterrows():
                    row_text = f"Row {idx + 1}: "
                    for col in df.columns:
                        row_text += f"{col}: {row[col]} | "
                    text_content += row_text.rstrip(" | ") + "\n"

                # Add summary if there are more rows
                if len(df) > 10:
                    text_content += f"\n... and {len(df) - 10} more rows\n"

                # Create document
                metadata = {
                    "sheet": sheet_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "parser": "llamaparse_fallback_pandas",
                }

                documents.append(Document(page_content=text_content, metadata=metadata))

            print(f"📊 Pandas fallback created {len(documents)} documents")
            return documents

        except Exception as pandas_error:
            print(f"❌ Pandas fallback also failed: {pandas_error}")
            raise Exception(
                f"Both LlamaParse and pandas processing failed. Please check your file and API keys."
            )

    def _process_pdf_file_llamaparse(self, file_path: str) -> List[Document]:
        """Process PDF file using LlamaParse with Unstructured fallback"""
        print(f"📄 Processing PDF file with LlamaParse: {file_path}")

        # Use LlamaParse for PDF processing
        if self.llama_parser:
            try:
                print(f"📄 Using LlamaParse for enhanced PDF processing: {file_path}")

                # Try LlamaParse (with minimal retry only if empty)
                llama_docs = self.llama_parser.load_data(file_path)

                if not llama_docs:
                    print(f"⚠️ LlamaParse returned empty result, trying once more...")
                    llama_docs = self.llama_parser.load_data(file_path)

                print(
                    f"📄 LlamaParse loaded {len(llama_docs) if llama_docs else 0} chunks from PDF"
                )

                # Convert LlamaParse documents to LangChain Document format
                documents = []
                for i, doc in enumerate(llama_docs):
                    # LlamaParse documents have 'text' attribute instead of 'page_content'
                    content = doc.text if hasattr(doc, "text") else str(doc)

                    # Skip empty documents
                    if not content or not content.strip():
                        continue

                    # Start with LlamaParse metadata if available
                    doc_metadata = (
                        doc.metadata.copy()
                        if hasattr(doc, "metadata") and doc.metadata
                        else {}
                    )

                    # Add our own metadata
                    doc_metadata.update(
                        {"source_index": i, "page": i, "parser": "llamaparse"}
                    )

                    documents.append(
                        Document(page_content=content.strip(), metadata=doc_metadata)
                    )

                if documents:
                    return documents

            except Exception as llama_error:
                print(f"⚠️ LlamaParse failed: {llama_error}")
                print("🔄 Falling back to Unstructured processing...")

        # Fallback to Unstructured processing if LlamaParse fails
        print("📄 Using Unstructured fallback for PDF processing")
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader

            # Load PDF with Unstructured
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()

            # Update metadata to indicate fallback
            for doc in documents:
                doc.metadata["parser"] = "llamaparse_fallback_unstructured"

            print(f"📄 Unstructured fallback created {len(documents)} documents")
            return documents

        except Exception as unstructured_error:
            print(f"❌ Unstructured fallback also failed: {unstructured_error}")
            raise Exception(
                f"Both LlamaParse and Unstructured processing failed. Please check your file and API keys."
            )

    def query(
        self,
        question: str,
        use_web_search: bool = False,
        max_results: Optional[int] = None,
    ) -> Dict:
        """
        Query the RAG system with quality enhancements

        Args:
            question: User's question
            use_web_search: Whether to augment with web search
            max_results: Maximum number of chunks to retrieve

        Returns:
            Dict with query results
        """
        start_time = time.time()

        try:
            print(f"❓ Processing query with LlamaParse RAG: {question}")

            # Analyze and enhance query
            query_analysis = self._analyze_query(question)
            enhanced_question = query_analysis.enhanced_query

            # Check cache first
            cache_key = f"llamaparse_query:{hash(enhanced_question)}"
            cached_result = self.get_fallback_cache(cache_key)
            if cached_result:
                print("✅ Returning cached result")
                return cached_result

            # Load vectorstore
            vectorstore = self.load_index()
            if vectorstore is None:
                return {
                    "status": "error",
                    "error": "No documents found. Please upload files first!",
                    "parser": "llamaparse",
                }

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": max_results or self.max_retrieval_results}
                ),
                return_source_documents=True,
            )

            # Generate answer with enhanced query
            print(f"🔍 Generating answer with LlamaParse RAG...")
            result = qa_chain.invoke({"query": enhanced_question})

            # Extract answer with better error handling
            if result and "result" in result:
                answer = result["result"]
                source_documents = result.get("source_documents", [])

                # Calculate metrics
                query_time = time.time() - start_time

                response = {
                    "status": "success",
                    "answer": answer,
                    "question": question,
                    "enhanced_question": enhanced_question,
                    "source_documents": [
                        doc.page_content[:200] + "..." for doc in source_documents
                    ],
                    "parser": "llamaparse",
                    "metrics": {
                        "query_time": query_time,
                        "source_count": len(source_documents),
                        "chunk_count": len(source_documents),
                    },
                }

                # Cache the result
                self.set_fallback_cache(cache_key, response, ttl=3600)

                print(f"✅ LlamaParse RAG query completed in {query_time:.2f}s")
                return response
            else:
                return {
                    "status": "error",
                    "error": "Failed to generate answer",
                    "parser": "llamaparse",
                }

        except Exception as e:
            print(f"❌ LlamaParse RAG query failed: {e}")
            return {"status": "error", "error": str(e), "parser": "llamaparse"}

    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze and enhance query for better retrieval"""
        try:
            # Simple keyword extraction
            keywords = [word.lower() for word in query.split() if len(word) > 3]

            # Simple intent detection
            intent = "general"
            if any(word in query.lower() for word in ["how many", "count", "number"]):
                intent = "counting"
            elif any(
                word in query.lower() for word in ["what is", "define", "explain"]
            ):
                intent = "definition"
            elif any(word in query.lower() for word in ["how", "process", "method"]):
                intent = "how_to"

            # Language detection (assume English for now)
            language = "english"

            # Confidence based on query length and keywords
            confidence = min(1.0, len(query.split()) / 10.0)

            # Generate enhanced query
            enhanced_query = self._enhance_query(query, keywords, intent, language)

            return QueryAnalysis(
                original_query=query,
                enhanced_query=enhanced_query,
                keywords=keywords,
                intent=intent,
                language=language,
                confidence=confidence,
                suggestions=[],
            )

        except Exception as e:
            print(f"⚠️ Query analysis failed: {e}")
            return QueryAnalysis(
                original_query=query,
                enhanced_query=query,
                keywords=[],
                intent="general",
                language="english",
                confidence=0.5,
                suggestions=[],
            )

    def _enhance_query(
        self, query: str, keywords: List[str], intent: str, language: str
    ) -> str:
        """Enhance query for better retrieval with rule-based fallback"""
        query_lower = query.lower()

        # Handle "first product" queries specifically
        if any(
            phrase in query_lower
            for phrase in [
                "first product",
                "first item",
                "first row",
                "row 1",
                "row one",
            ]
        ):
            # Enhance to be more specific about finding the first product in the data
            enhanced = (
                f"{query} (find the product name from the first row of data, Row 1)"
            )
            print(f"🔍 Enhanced 'first product' query: {enhanced}")
            return enhanced

        # Handle "product name" queries
        if "product name" in query_lower and "first" in query_lower:
            enhanced = f"{query} (locate the product name from Row 1 of the dataset)"
            print(f"🔍 Enhanced 'first product name' query: {enhanced}")
            return enhanced

        try:
            # Try to enhance with LLM
            prompt = f"""
            Enhance this query for better document retrieval:
            Original: "{query}"
            Intent: {intent}
            Keywords: {', '.join(keywords)}
            
            Return only the enhanced query, no explanations.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            enhanced = response.content.strip()

            # Validate enhanced query
            if len(enhanced) > 10 and enhanced != query:
                return enhanced

        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {e}")

        # Fallback: return original query
        return query

    def load_index(self):
        """Load FAISS index from disk"""
        try:
            index_path = self.base_storage_dir / "faiss_index"
            if index_path.exists():
                print(f"📁 Loading FAISS index from {index_path}")
                vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ Loaded FAISS index with {vectorstore.index.ntotal} vectors")
                return vectorstore
            else:
                print(f"📁 No existing index found at {index_path}")
                return None
        except Exception as e:
            print(f"⚠️ Failed to load index: {e}")
            return None

    def save_index(self, vectorstore):
        """Save FAISS index to disk"""
        try:
            index_path = self.base_storage_dir / "faiss_index"
            vectorstore.save_local(str(index_path))
            print(f"💾 Saved FAISS index to {index_path}")
        except Exception as e:
            print(f"⚠️ Failed to save index: {e}")

    def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            vectorstore = self.load_index()
            if vectorstore:
                return {
                    "status": "success",
                    "total_vectors": vectorstore.index.ntotal,
                    "index_size_mb": self._get_index_size(),
                    "parser": "llamaparse",
                }
            else:
                return {
                    "status": "no_index",
                    "total_vectors": 0,
                    "index_size_mb": 0,
                    "parser": "llamaparse",
                }
        except Exception as e:
            return {"status": "error", "error": str(e), "parser": "llamaparse"}

    def _get_index_size(self) -> float:
        """Get index file size in MB"""
        try:
            index_path = self.base_storage_dir / "faiss_index"
            if index_path.exists():
                size_bytes = sum(
                    f.stat().st_size for f in index_path.rglob("*") if f.is_file()
                )
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0
