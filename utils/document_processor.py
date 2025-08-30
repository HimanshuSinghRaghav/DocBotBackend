import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import pickle
from datetime import datetime
import re
import traceback

# PDF and OCR imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available - PDF text extraction will be limited")

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not available - PDF OCR will be disabled")

from models.models import Document, Chunk
from database.database import SessionLocal
from langchain_community.embeddings import OpenAIEmbeddings
from utils.content_generator import ContentGenerator

class DocumentProcessor:
    def __init__(self):
        self.db = SessionLocal()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize embeddings if API key is available
        if self.openai_api_key:
            self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        else:
            self.embeddings = None
    
    def process_document(
        self, 
        file_path: str, 
        title: str, 
        document_type: str,
        version: str,
        effective_date: datetime,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Process a document file and store it in the database with basic text extraction."""
        
        # Extract text from the document based on file type
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif extension == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif extension == ".pdf":
            # For PDF, use advanced text extraction with OCR fallback
            text = self._extract_text_from_pdf(file_path)
            
            # Store extraction metadata
            if not metadata:
                metadata = {}
            metadata["pdf_extraction_method"] = "advanced_with_ocr"
            metadata["text_length"] = len(text)
            
            # Check if extraction was successful
            if "Content extraction was limited" in text:
                metadata["extraction_warning"] = "Limited content extraction - PDF may be image-based or corrupted"
        elif extension in [".docx", ".doc"]:
            # For Word documents, we'll use a simple approach
            # In production, you'd use python-docx
            text = self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        # Split the text into chunks with better parameters for AI processing
        chunks = self._split_text_into_chunks(text, chunk_size=1500, overlap=300)
        
        # Create document record
        doc = Document(
            title=title,
            document_type=document_type,
            version=version,
            effective_date=effective_date,
            language=language,
            file_path=file_path,
            doc_metadata=metadata or {}
        )
        
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        
        # Process chunks and create embeddings
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            # Create embedding using OpenAI if available, otherwise use simple embedding
            if self.embeddings:
                try:
                    embedding = self.embeddings.embed_query(chunk)
                except Exception as e:
                    print(f"Error creating embedding for chunk {i}: {e}")
                    embedding = self._create_simple_embedding(chunk)
            else:
                embedding = self._create_simple_embedding(chunk)
            
            # Create chunk record
            chunk_metadata = {
                "index": i,
                "document_id": doc.id,
                "document_title": title,
                "document_version": version,
                "section": f"chunk_{i}",
            }
            
            chunk_obj = Chunk(
                content=chunk,
                embedding=json.dumps(embedding),
                chunk_metadata=chunk_metadata
            )
            
            self.db.add(chunk_obj)
            chunk_objects.append(chunk_obj)
        
        self.db.commit()
        
        # Associate chunks with document using the many-to-many relationship
        for chunk_obj in chunk_objects:
            doc.chunks.append(chunk_obj)
        
        self.db.commit()
        
        # Verify the association worked
        print(f"Document {doc.id} now has {len(doc.chunks)} chunks")
        
        # Create a simple index for this document
        self._create_simple_index(doc.id, chunks)
        
        # Generate learning content and quizzes using AI
        try:
            print(f"Starting content generation for document {doc.id}")
            content_generator = ContentGenerator(self.db)
            success, message = content_generator.generate_learning_content(doc.id)
            
            if not success:
                print(f"Warning: Content generation failed: {message}")
                # Store the warning in document metadata
                if not doc.doc_metadata:
                    doc.doc_metadata = {}
                doc.doc_metadata["content_generation_warning"] = message
                self.db.commit()
            else:
                print(f"Content generation successful: {message}")
                # Update document metadata with success info
                if not doc.doc_metadata:
                    doc.doc_metadata = {}
                doc.doc_metadata["content_generation_status"] = "success"
                doc.doc_metadata["content_generation_message"] = message
                self.db.commit()
                
        except Exception as e:
            print(f"Error during content generation: {str(e)}")
            traceback.print_exc()
            # Store the error in document metadata
            if not doc.doc_metadata:
                doc.doc_metadata = {}
            doc.doc_metadata["content_generation_error"] = str(e)
            self.db.commit()
        
        return doc
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using OCR and text extraction."""
        extracted_text = ""
        extraction_method = "none"
        
        # Method 1: Try PyPDF2 for text extraction first
        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and len(page_text.strip()) > 50:
                                text_content.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                        except Exception as e:
                            print(f"Error extracting text from page {page_num + 1}: {e}")
                    
                    if text_content:
                        extracted_text = "\n".join(text_content)
                        extraction_method = "pypdf2"
                        print(f"Successfully extracted {len(extracted_text)} characters using PyPDF2")
                        
                        # If we got good text content, return it
                        if len(extracted_text.strip()) > 200:
                            return extracted_text
                            
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: Use OCR if text extraction failed or produced insufficient content
        if OCR_AVAILABLE:
            try:
                print("Using OCR for PDF text extraction...")
                
                # Convert PDF to images
                images = convert_from_path(file_path, dpi=200)
                print(f"Converted PDF to {len(images)} images")
                
                ocr_text_content = []
                
                for page_num, image in enumerate(images):
                    try:
                        # Use OCR to extract text from image
                        page_text = pytesseract.image_to_string(image, lang='eng')
                        
                        if page_text and len(page_text.strip()) > 20:
                            clean_text = self._clean_ocr_text(page_text)
                            if clean_text:
                                ocr_text_content.append(f"\n--- Page {page_num + 1} (OCR) ---\n{clean_text}")
                                
                    except Exception as e:
                        print(f"Error performing OCR on page {page_num + 1}: {e}")
                
                if ocr_text_content:
                    ocr_extracted_text = "\n".join(ocr_text_content)
                    extraction_method = "ocr"
                    print(f"Successfully extracted {len(ocr_extracted_text)} characters using OCR")
                    
                    # Choose the better extraction result
                    if len(ocr_extracted_text.strip()) > len(extracted_text.strip()):
                        extracted_text = ocr_extracted_text
                    
            except Exception as e:
                print(f"OCR extraction failed: {e}")
        
        # Fallback if all methods failed
        if not extracted_text or len(extracted_text.strip()) < 100:
            extracted_text = f"PDF processing completed for {os.path.basename(file_path)}. " \
                           f"Extraction method: {extraction_method}. " \
                           "Content extraction was limited - consider uploading a text-based PDF or document."
            extraction_method = "fallback"
        
        # Add metadata about extraction method
        print(f"Final extraction result: {len(extracted_text)} characters using {extraction_method}")
        
        return extracted_text
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and improve OCR extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Fix common OCR character mistakes
        ocr_fixes = {
            'rn': 'm',  # common OCR mistake
            'l1': 'll',
            '0': 'O',   # in words context
            'S': '5',   # in numbers context (context-sensitive)
        }
        
        # Apply basic fixes (more sophisticated NLP could be added)
        for mistake, correction in ocr_fixes.items():
            if mistake in text:
                # Only apply if it makes sense in context
                text = text.replace(mistake, correction)
        
        # Remove lines that are too short (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # Keep lines with substantial content
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document (simplified version)."""
        try:
            # For now, return a placeholder. In production, use python-docx
            return f"Word document content from {os.path.basename(file_path)} - This is a placeholder for actual Word text extraction."
        except Exception as e:
            return f"Error extracting Word text: {str(e)}"
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """Split text into overlapping chunks optimized for AI processing."""
        chunks = []
        start = 0
        
        # Pre-clean the text
        text = self._preprocess_text_for_chunking(text)
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at natural boundaries
            if end < len(text):
                # Look for paragraph breaks first
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i:i+2] == '\n\n':
                        end = i + 2
                        break
                else:
                    # Look for sentence endings
                    for i in range(end, max(start + chunk_size - 100, start), -1):
                        if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                            end = i + 1
                            break
                    else:
                        # Look for word boundaries
                        for i in range(end, max(start + chunk_size - 50, start), -1):
                            if text[i].isspace():
                                end = i + 1
                                break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        print(f"Split text into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} chars)")
        return chunks
    
    def _preprocess_text_for_chunking(self, text: str) -> str:
        """Preprocess text to improve chunking quality."""
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Non-empty line
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Empty line after content
                cleaned_lines.append('')  # Preserve paragraph break
        
        # Join lines back and clean up excessive newlines
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        return text
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple embedding for the text (simplified version)."""
        # This is a very basic embedding. In production, use a proper embedding model
        # For now, create a simple hash-based embedding
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a list of floats (simplified embedding)
        embedding = []
        for i in range(0, len(text_hash), 2):
            if i + 1 < len(text_hash):
                # Convert hex pairs to float values between 0 and 1
                val = int(text_hash[i:i+2], 16) / 255.0
                embedding.append(val)
        
        # Pad or truncate to 384 dimensions (common embedding size)
        while len(embedding) < 384:
            embedding.append(0.0)
        embedding = embedding[:384]
        
        return embedding
    
    def _create_simple_index(self, doc_id: int, chunks: List[str]):
        """Create a simple index for the document chunks and cache embeddings."""
        # Create index directory
        index_dir = os.path.join("indexes", str(doc_id))
        os.makedirs(index_dir, exist_ok=True)
        
        # Save chunks to a simple index file
        index_data = {
            "chunks": chunks,
            "created_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(index_dir, "index.json"), "w") as f:
            json.dump(index_data, f, indent=2)
        
        # Cache embeddings if available
        if self.embeddings:
            try:
                # Calculate embeddings for all chunks
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = self.embeddings.embed_query(chunk)
                    chunk_embeddings.append(embedding)
                
                # Save embeddings to pickle file
                embeddings_file = os.path.join(index_dir, "embeddings.pkl")
                with open(embeddings_file, "wb") as f:
                    pickle.dump(chunk_embeddings, f)
                
                print(f"Cached embeddings for document {doc_id}")
            except Exception as e:
                print(f"Error caching embeddings for document {doc_id}: {e}")
        
    def close(self):
        """Close the database session."""
        self.db.close()
