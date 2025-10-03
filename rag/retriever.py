# retriever.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from config import PERSIST_DIRECTORY, ENABLE_HYBRID_SEARCH, HYBRID_ALPHA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, persist_directory=PERSIST_DIRECTORY,
                 collection_name=None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            if self.collection_name is None:
                from config import generate_collection_name
                self.collection_name = generate_collection_name()

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
                doc_id = f"{doc['metadata'].get('filename', 'unknown')}_{doc['metadata'].get('global_chunk_id', i)}"
                ids.append(doc_id)
                embeddings.append(doc['embedding'])
                
                metadata = {}
                for key, value in doc['metadata'].items():
                    if key == 'page_number' and value is not None:
                        metadata[key] = str(value)
                    else:
                        metadata[key] = str(value) if value is not None else ''
                metadatas.append(metadata)
                documents_text.append(doc['content'])

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )

            logger.info(f"Added {len(documents)} documents to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            raise

    def search(self, query_embedding, query_text=None, top_k=5, use_hybrid=True):
        try:
            if use_hybrid and query_text and ENABLE_HYBRID_SEARCH:
                return self._hybrid_search(query_embedding, query_text, top_k)
            else:
                return self._semantic_search(query_embedding, top_k)

        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []

    def _semantic_search(self, query_embedding, top_k):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format_search_results(results)

    def _hybrid_search(self, query_embedding, query_text, top_k):
        # Semantic search
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # Lấy nhiều kết quả hơn để kết hợp
        )

        # Keyword search
        keyword_results = self._keyword_search(query_text, top_k * 2)

        # Kết hợp kết quả
        combined_results = self._combine_results(
            semantic_results, keyword_results, top_k
        )

        return combined_results

    def _keyword_search(self, query_text, top_k):
        try:
            all_docs = self.collection.get(limit=1000)  # Adjust limit as needed
            
            if not all_docs['documents']:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

            keywords = self._extract_keywords(query_text)
            if not keywords:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

            # Tìm documents chứa ít nhất một từ khóa
            matching_indices = []
            for i, doc_content in enumerate(all_docs['documents']):
                doc_lower = doc_content.lower()
                for keyword in keywords:
                    if keyword in doc_lower:
                        matching_indices.append(i)
                        break  # Chỉ cần khớp một từ khóa

            # Lấy kết quả matching
            if matching_indices:
                matching_docs = [all_docs['documents'][i] for i in matching_indices]
                matching_metadatas = [all_docs['metadatas'][i] for i in matching_indices]
                
                # Sắp xếp theo số từ khóa khớp
                scored_docs = []
                for doc, metadata in zip(matching_docs, matching_metadatas):
                    score = sum(1 for keyword in keywords if keyword in doc.lower())
                    scored_docs.append((score, doc, metadata))
                
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_scored = scored_docs[:top_k]
                
                documents = [doc for _, doc, _ in top_scored]
                metadatas = [metadata for _, _, metadata in top_scored]
                
                formatted_results = {
                    'documents': [documents],
                    'metadatas': [metadatas],
                    'distances': [[0.0] * len(documents)]
                }
            else:
                formatted_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

            return formatted_results

        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    def _extract_keywords(self, text):
        # Loại bỏ stopwords đơn giản và trích xuất từ khóa
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        return keywords[:5]  # Giới hạn số từ khóa

    def _combine_results(self, semantic_results, keyword_results, top_k):
        semantic_docs = self._format_search_results(semantic_results)
        keyword_docs = self._format_search_results(keyword_results)

        # Kết hợp và xếp hạng lại
        all_docs = {}
        
        # Thêm semantic results với trọng số
        for doc in semantic_docs:
            doc_id = doc['metadata'].get('filename', '') + str(doc['metadata'].get('global_chunk_id', ''))
            all_docs[doc_id] = {
                'doc': doc,
                'score': HYBRID_ALPHA * (1 - (doc.get('distance', 0) if doc.get('distance') else 0))
            }

        # Thêm keyword results với trọng số
        for doc in keyword_docs:
            doc_id = doc['metadata'].get('filename', '') + str(doc['metadata'].get('global_chunk_id', ''))
            if doc_id in all_docs:
                all_docs[doc_id]['score'] += (1 - HYBRID_ALPHA) * 0.8  # Trọng số cho keyword match
            else:
                all_docs[doc_id] = {
                    'doc': doc,
                    'score': (1 - HYBRID_ALPHA) * 0.8
                }

        # Sắp xếp theo điểm và lấy top_k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x['score'], reverse=True)
        final_results = [item['doc'] for item in sorted_docs[:top_k]]

        return final_results

    def _format_search_results(self, results):
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                }
                formatted_results.append(result)

        return formatted_results

    # Các phương thức khác giữ nguyên...
    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")

    def get_collection_info(self):
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