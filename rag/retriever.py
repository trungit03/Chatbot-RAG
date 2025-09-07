import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from config import PERSIST_DIRECTORY, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorRetriever:
    """Vector database retriever using ChromaDB"""

    def __init__(self, persist_directory: str = PERSIST_DIRECTORY,
                 collection_name: str = COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        if not documents:
            logger.warning("No documents to add")
            return

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []

            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = f"{doc['metadata'].get('filename', 'unknown')}_{doc['metadata'].get('chunk_id', i)}"
                ids.append(doc_id)

                # Add embedding
                embeddings.append(doc['embedding'])

                # Add metadata (ChromaDB requires string values)
                metadata = {}
                for key, value in doc['metadata'].items():
                    metadata[key] = str(value)
                metadatas.append(metadata)

                # Add document text
                documents_text.append(doc['content'])

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )

            logger.info(f"Added {len(documents)} documents to vector database")

        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Get all IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info("Cleared all documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")