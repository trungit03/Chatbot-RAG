import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import pymupdf
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from config import SUPPORTED_EXTENSIONS
import io
import psutil
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS
        self.table_detector = None
        self._initialize_table_detector()

    def _initialize_table_detector(self):
        try:
            import cv2
            self.table_detector = cv2
            logger.info("Table detector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize table detector: {str(e)}")

    def load_document(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        try:
            content_data = self._load_enhanced_pdf(file_path)

            return {
                'content': content_data['formatted_text'],
                'pages': content_data['pages'],
                'tables': content_data['tables'],
                'images': content_data['images'],
                'metadata': {
                    'filename': file_path.name,
                    'file_path': str(file_path),
                    'file_type': extension,
                    'file_size': file_path.stat().st_size,
                    'page_count': content_data['page_count'],
                    'has_tables': len(content_data['tables']) > 0,
                    'has_images': len(content_data['images']) > 0
                }
            }
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def _load_enhanced_pdf(self, file_path):
        try:
            doc = pymupdf.open(file_path)
            text_content = []
            page_contents = []
            all_tables = []
            all_images = []
            page_count = len(doc)

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                
                # Trích xuất văn bản thông thường
                page_text = page.get_text()
                
                # Trích xuất và xử lý hình ảnh
                page_images = self._extract_images(page, page_num, file_path)
                all_images.extend(page_images)
                
                # Trích xuất và nhận dạng table
                page_tables = self._extract_tables(page, page_num, file_path)
                all_tables.extend(page_tables)
                
                # Kết hợp tất cả nội dung
                combined_content = page_text
                
                # Thêm mô tả table vào nội dung
                if page_tables:
                    table_descriptions = "\n".join([
                        f"[Table {i+1}]: {table.get('description', 'Extracted table content')}"
                        for i, table in enumerate(page_tables)
                    ])
                    combined_content += f"\n\n--- TABLES ON PAGE {page_num + 1} ---\n{table_descriptions}"
                
                # Thêm mô tả hình ảnh vào nội dung
                if page_images:
                    image_descriptions = "\n".join([
                        f"[Image {i+1}]: {img.get('description', 'Extracted image')}"
                        for i, img in enumerate(page_images)
                    ])
                    combined_content += f"\n\n--- IMAGES ON PAGE {page_num + 1} ---\n{image_descriptions}"

                if combined_content.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{combined_content}")
                    page_contents.append({
                        'page_number': page_num + 1,
                        'content': combined_content,
                        'has_tables': len(page_tables) > 0,
                        'has_images': len(page_images) > 0
                    })

            doc.close()

            return {
                'formatted_text': '\n\n'.join(text_content),
                'pages': page_contents,
                'tables': all_tables,
                'images': all_images,
                'page_count': page_count
            }

        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            raise

    def _extract_images(self, page, page_num, file_path):
        images = []
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = pymupdf.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # Kiểm tra CMYK
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    
                    img_data = pix.tobytes("png")
                    pix = None
                    
                    # Sử dụng OCR để mô tả hình ảnh
                    description = self._describe_image_with_ocr(img_data)
                    
                    images.append({
                        'page_number': page_num + 1,
                        'image_index': img_index,
                        'description': description,
                        'size': len(img_data),
                        'type': 'embedded'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error extracting images from page {page_num}: {str(e)}")
            
        return images

    def _describe_image_with_ocr(self, image_data):
        try:
            # Chuyển đổi dữ liệu ảnh thành PIL Image
            image = Image.open(io.BytesIO(image_data))

            text = pytesseract.image_to_string(image, lang='eng+vie')
            
            if text.strip():
                return f"OCR text: {text.strip()[:200]}..."
            else:
                return "Image contains no detectable text"
                
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}")
            return "Image (content not extracted)"

    def _extract_tables(self, page, page_num, file_path):
        tables = []
        try:
            text = page.get_text("dict")
            
            for block in text.get("blocks", []):
                if block.get("type") == 1:  # Block type 1 thường là image/table
                    bbox = block.get("bbox", [])
                    if self._is_likely_table(bbox, block):
                        table_text = self._extract_table_text(block)
                        if table_text:
                            tables.append({
                                'page_number': page_num + 1,
                                'bbox': bbox,
                                'content': table_text,
                                'type': 'structured',
                                'description': f"Structured table: {table_text[:100]}..."
                            })
            
            cv_tables = self._detect_tables_with_cv(page, page_num)
            tables.extend(cv_tables)
            
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {str(e)}")
            
        return tables

    def _is_likely_table(self, bbox, block):
        if len(bbox) != 4:
            return False
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width > 100 and height > 50  # Kích thước tối thiểu

    def _extract_table_text(self, block) -> str:
        try:
            lines = []
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            lines.append(text)
            return " | ".join(lines) if lines else ""
        except Exception:
            return ""

    def _detect_tables_with_cv(self, page, page_num):
        tables = []
        try:
            # Chuyển trang PDF thành ảnh
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Chuyển đổi sang OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Phát hiện đường thẳng (để tìm table grid)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 10:
                table_text = self._ocr_table_area(img)
                tables.append({
                    'page_number': page_num + 1,
                    'type': 'detected_grid',
                    'content': table_text,
                    'description': f"Grid-based table: {table_text[:100]}..."
                })
                
        except Exception as e:
            logger.warning(f"CV table detection failed: {str(e)}")
            
        return tables

    def _ocr_table_area(self, img):
        try:
            # Sử dụng OCR để trích xuất text từ vùng ảnh
            text = pytesseract.image_to_string(img)
            return text.strip() if text.strip() else "Table content (no text extracted)"
        except Exception:
            return "Table content (OCR failed)"

    def load_documents(self, directory_path: str) -> List[Dict[str, Any]]:
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
                logger.info(f"Loaded PDF: {file_path.name} "
                          f"({doc['metadata']['page_count']} pages, "
                          f"{len(doc['tables'])} tables, "
                          f"{len(doc['images'])} images)")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {str(e)}")
        return documents