# chatbot.py
import json
import os
from datetime import datetime
import logging
from pathlib import Path

from utils.reranker import Reranker
from config import ENABLE_RERANKER, RERANKER_MODEL, ENABLE_HYBRID_SEARCH
from utils import DocumentLoader, TextProcessor, EmbeddingManager
from rag.retriever import VectorRetriever
from rag.llm import OllamaLLM
from config import CHAT_HISTORY_DIR, MAX_CHAT_HISTORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self, collection_name=None):
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.retriever = VectorRetriever(collection_name=collection_name)
        self.llm = OllamaLLM()
        self.reranker = Reranker(RERANKER_MODEL) if ENABLE_RERANKER else None
        self.chat_history = []
        self.conversation_context = []
        self.session_id = None
        self.is_initialized = False

        logger.info("RAG Chatbot initialized successfully")

    def load_documents(self, document_path):
        try:
            logger.info(f"Loading documents from: {document_path}")

            if os.path.isfile(document_path):
                documents = [self.document_loader.load_document(document_path)]
            else:
                documents = self.document_loader.load_documents(document_path)

            if not documents:
                logger.warning("No documents found to load")
                return False

            logger.info("Processing documents into chunks...")
            chunks = self.text_processor.process_documents(documents)

            if not chunks:
                logger.warning("No chunks created from documents")
                return False

            logger.info("Generating embeddings...")
            embedded_chunks = self.embedding_manager.embed_documents(chunks)

            logger.info("Adding documents to vector database...")
            self.retriever.add_documents(embedded_chunks)

            self.is_initialized = True
            logger.info(f"Successfully loaded {len(documents)} documents with {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def _build_conversation_context(self, current_query, window_size = 3):
        if not self.chat_history:
            return current_query

        # Lấy các exchange gần nhất
        recent_exchanges = self.chat_history[-window_size:]
        
        context_parts = []
        for exchange in recent_exchanges:
            context_parts.append(f"User: {exchange.get('human', '')}")
            context_parts.append(f"Assistant: {exchange.get('assistant', '')}")
        
        context_parts.append(f"Current Question: {current_query}")
        return "\n".join(context_parts)
    
    def chat(self, user_message, top_k=5):
        try:
            if not self.is_initialized:
                return "Please load documents first before asking questions."

            conversation_context = self._build_conversation_context(user_message)
            query_embedding = self.embedding_manager.embed_text(user_message)
            relevant_docs = self.retriever.search(
                query_embedding, 
                query_text=user_message,  
                top_k=top_k * 2,  # Lấy nhiều hơn để re-rank
                use_hybrid=ENABLE_HYBRID_SEARCH
            )

            if self.reranker and self.reranker.is_available():
                relevant_docs = self.reranker.rerank(user_message, relevant_docs, top_k)
            else:
                relevant_docs = relevant_docs[:top_k]

            # Prepare context with both content and metadata 
            context = []
            for doc in relevant_docs:
                # Ensure metadata is properly handled
                metadata = doc.get('metadata', {})
                context.append({
                    'content': doc.get('content', ''),
                    'metadata': metadata
                })

            response = self.llm.generate_response(
                prompt=user_message,
                context=context,
                chat_history=self.chat_history,
                conversation_context=conversation_context
            )

            self._update_chat_history(user_message, response, relevant_docs)

            return response

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "Sorry, I encountered an error while processing your question."

    def stream_chat(self, user_message, top_k=5):
        try:
            if not self.is_initialized:
                yield "Please load documents first before asking questions."
                return

            conversation_context = self._build_conversation_context(user_message)
            query_embedding = self.embedding_manager.embed_text(user_message)
            relevant_docs = self.retriever.search(
                query_embedding,
                query_text=user_message,
                top_k=top_k * 2,
                use_hybrid=ENABLE_HYBRID_SEARCH
            )

            if self.reranker and self.reranker.is_available():
                relevant_docs = self.reranker.rerank(user_message, relevant_docs, top_k)
            else:
                relevant_docs = relevant_docs[:top_k]

            # Prepare context with both content and metadata - FIXED
            context = []
            for doc in relevant_docs:
                context.append({
                    'content': doc.get('content', ''),
                    'metadata': doc.get('metadata', {})
                })

            full_response = ""
            for chunk in self.llm.stream_response(
                    prompt=user_message,
                    context=context,
                    chat_history=self.chat_history,
                    conversation_context=conversation_context
            ):
                full_response += chunk
                yield chunk

            self._update_chat_history(user_message, full_response, relevant_docs)

        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            yield f"Error: {str(e)}"

    def _update_chat_history(self, user_message, assistant_response, relevant_docs):
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'human': user_message,
            'assistant': assistant_response,
            'sources': [doc.get('metadata', {}) for doc in relevant_docs]
        }

        self.chat_history.append(exchange)

        if len(self.chat_history) > MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-MAX_CHAT_HISTORY:]

    def clear_chat_history(self):
        self.chat_history = []
        logger.info("Chat history cleared")

    def save_chat_history(self, filename=None):
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.json"

            filepath = Path(CHAT_HISTORY_DIR) / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)

            logger.info(f"Chat history saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return None

    def load_chat_history(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.chat_history = json.load(f)

            logger.info(f"Chat history loaded from: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            return False

    def get_database_info(self):
        info = self.retriever.get_collection_info()
        info['is_initialized'] = self.is_initialized
        info['chat_history_length'] = len(self.chat_history)
        try:
            info['embedding_dimension'] = self.embedding_manager.get_embedding_dimension()
        except Exception:
            info['embedding_dimension'] = None
        return info

    def clear_database(self):
        self.retriever.clear_collection()
        self.is_initialized = False
        logger.info("Database cleared")

    def get_relevant_sources(self, user_message, top_k=5):
        try:
            if not self.is_initialized:
                return []

            query_embedding = self.embedding_manager.embed_text(user_message)
            relevant_docs = self.retriever.search(query_embedding, top_k=top_k)

            return relevant_docs

        except Exception as e:
            logger.error(f"Error getting relevant sources: {str(e)}")
            return []