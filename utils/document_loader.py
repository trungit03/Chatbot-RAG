import os
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import logging
from config import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS

    def load_document(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}. Only PDF files are supported.")

        try:
            content = self._load_pdf(file_path)

            return {
                'content': content,
                'metadata': {
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'file_type': extension,
                    'file_size': file_path.stat().st_size,
                    'page_count': self._get_pdf_page_count(file_path)
                }
            }
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def load_documents(self, directory_path):
        directory_path = Path(directory_path)
        documents = []

        if not directory_path.exists():
            logger.warning(f"Directory not found: {directory_path}")
            return documents

        pdf_files = list(directory_path.rglob('*.pdf'))

        if not pdf_files:
            logger.info(f"No PDF files found in {directory_path}")
            return documents

        for file_path in pdf_files:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                logger.info(f"Loaded PDF: {file_path.name} ({doc['metadata']['page_count']} pages)")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {str(e)}")

        logger.info(f"Successfully loaded {len(documents)} PDF documents")
        return documents

    def _load_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")

                if not text_content:
                    raise ValueError("No text could be extracted from the PDF")

                return '\n\n'.join(text_content)

        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            raise

    def _get_pdf_page_count(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception:
            return 0

    def validate_pdf(self, file_path):
        try:
            file_path = Path(file_path)
            if not file_path.exists() or file_path.suffix.lower() != '.pdf':
                return False

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    pdf_reader.pages[0].extract_text()
                return True
        except Exception:
            return False