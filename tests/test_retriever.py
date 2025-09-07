import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import VectorRetriever
from utils.embeddings import EmbeddingManager


class TestVectorRetriever(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.retriever = VectorRetriever(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )
        self.embedding_manager = EmbeddingManager()

        self.sample_docs = [
            {
                'content': 'This is a document about artificial intelligence and machine learning.',
                'metadata': {'filename': 'ai_doc.txt', 'chunk_id': 0},
                'embedding': self.embedding_manager.embed_text(
                    'This is a document about artificial intelligence and machine learning.')
            },
            {
                'content': 'Python is a popular programming language for data science.',
                'metadata': {'filename': 'python_doc.txt', 'chunk_id': 0},
                'embedding': self.embedding_manager.embed_text(
                    'Python is a popular programming language for data science.')
            },
            {
                'content': 'Natural language processing is a subfield of AI.',
                'metadata': {'filename': 'nlp_doc.txt', 'chunk_id': 0},
                'embedding': self.embedding_manager.embed_text('Natural language processing is a subfield of AI.')
            }
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_documents(self):
        self.retriever.add_documents(self.sample_docs)

        info = self.retriever.get_collection_info()
        self.assertEqual(info['document_count'], 3)

    def test_search(self):
        self.retriever.add_documents(self.sample_docs)

        query_embedding = self.embedding_manager.embed_text("artificial intelligence")
        results = self.retriever.search(query_embedding, top_k=2)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)

        self.assertIn('artificial intelligence', results[0]['content'].lower())

    def test_empty_search(self):
        query_embedding = self.embedding_manager.embed_text("test query")
        results = self.retriever.search(query_embedding)

        self.assertEqual(len(results), 0)

    def test_clear_collection(self):
        self.retriever.add_documents(self.sample_docs)

        info = self.retriever.get_collection_info()
        self.assertEqual(info['document_count'], 3)

        self.retriever.clear_collection()

        info = self.retriever.get_collection_info()
        self.assertEqual(info['document_count'], 0)

    def test_get_collection_info(self):
        info = self.retriever.get_collection_info()

        self.assertIn('name', info)
        self.assertIn('document_count', info)
        self.assertIn('persist_directory', info)
        self.assertEqual(info['name'], 'test_collection')


if __name__ == '__main__':
    unittest.main()