import logging
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name = RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            # Fallback to a smaller model if available
            try:
                self.model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                self.model = CrossEncoder(self.model_name)
                logger.info("Fallback reranker model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback reranker model: {str(e2)}")
                self.model = None

    def rerank(self, query, documents, top_k = 5):
        if not self.model or not documents:
            return documents[:top_k]

        try:
            # Chuẩn bị cặp query-document cho re-ranker
            pairs = [(query, doc['content']) for doc in documents]
            
            # Dự đoán điểm relevance
            scores = self.model.predict(pairs)
            
            # Kết hợp điểm với documents
            scored_docs = list(zip(scores, documents))
            
            # Sắp xếp theo điểm giảm dần
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Lấy top_k documents
            reranked_docs = [doc for score, doc in scored_docs[:top_k]]
            
            logger.info(f"Reranked {len(documents)} documents to top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return documents[:top_k]

    def is_available(self):
        return self.model is not None