import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from config import PERSIST_DIRECTORY, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, persist_directory = PERSIST_DIRECTORY,
                 collection_name = COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents):
        if not documents:
            logger.warning("No documents to add")
            return

        try:
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []

            for i, doc in enumerate(documents):
                doc_id = f"{doc['metadata'].get('filename', 'unknown')}_{doc['metadata'].get('chunk_id', i)}"
                ids.append(doc_id)

                embeddings.append(doc['embedding'])

                metadata = {}
                for key, value in doc['metadata'].items():
                    metadata[key] = str(value)
                metadatas.append(metadata)

                documents_text.append(doc['content'])

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

    def search(self, query_embedding, top_k = 5):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

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
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
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
        try:
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info("Cleared all documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")