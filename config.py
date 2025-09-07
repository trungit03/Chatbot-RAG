import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORDB_DIR = DATA_DIR / "vectordb"
CHAT_HISTORY_DIR = DATA_DIR / "chat_history"

# Create directories if they don't exist
for dir_path in [DATA_DIR, DOCUMENTS_DIR, VECTORDB_DIR, CHAT_HISTORY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
TEMPERATURE = 0.7

# Vector database settings
PERSIST_DIRECTORY = str(VECTORDB_DIR)
COLLECTION_NAME = "rag_documents"

# Chat settings
MAX_CHAT_HISTORY = 10

# Document processing settings - PDF only
SUPPORTED_EXTENSIONS = ['.pdf']
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
#EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"