import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, chunk_size = 1000, chunk_overlap = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)

        text = re.sub(r'([.!?]){2,}', r'\1', text)

        return text.strip()

    def chunk_text(self, text, metadata = None):
        if not text or not text.strip():
            return []

        cleaned_text = self.clean_text(text)

        chunks = self.text_splitter.split_text(cleaned_text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': i,
                'chunk_size': len(chunk),
                **(metadata or {})
            }

            processed_chunks.append({
                'content': chunk,
                'metadata': chunk_metadata
            })

        logger.info(f"Created {len(processed_chunks)} chunks from text")
        return processed_chunks

    def process_documents(self, documents):
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_text(doc['content'], doc['metadata'])
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Error processing document {doc.get('metadata', {}).get('filename', 'unknown')}: {str(e)}")

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def extract_keywords(self, text, max_keywords = 10):
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]