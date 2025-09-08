from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        texts = [doc['content'] for doc in documents]

        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embed_texts(texts)

        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding

        logger.info("Embeddings added to all documents")
        return documents

    def similarity_search(self, query_embedding: List[float],
                          document_embeddings: List[List[float]],
                          top_k: int = 5) -> List[int]:
        query_embedding = np.array(query_embedding)
        document_embeddings = np.array(document_embeddings)

        similarities = np.dot(document_embeddings, query_embedding) / (
                np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()