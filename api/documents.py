from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import uuid
from datetime import datetime
import traceback

from database.database import get_db
from models.models import Document, Chunk
from schemas.document_schema import DocumentCreate, DocumentResponse, DocumentQuery
from utils.document_processor import DocumentProcessor
from utils.rag_engine import RAGEngine

router = APIRouter()

# Directory to store uploaded documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.doc'}

def process_document_background(
    document_id: int,
    file_path: str,
    title: str,
    document_type: str,
    version: str,
    effective_date: datetime,
    language: str,
    metadata: dict
):
    """
    Background task to process document after upload.
    This includes OCR, chunking, embedding generation, and learning content creation.
    """
    from database.database import SessionLocal
    from utils.document_processor import DocumentProcessor
    
    db = SessionLocal()
    try:
        # Get the document record
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"Document {document_id} not found for background processing")
            return
        
        # Update status to processing
        document.processing_status = "processing"
        document.processing_progress = 10
        document.processing_message = "Starting document processing..."
        document.processing_started_at = datetime.utcnow()
        db.commit()
        
        print(f"Starting background processing for document {document_id}: {title}")
        
        # Step 1: Process document (OCR, chunking, embeddings)
        document.processing_progress = 20
        document.processing_message = "Extracting text from document..."
        db.commit()
        
        processor = DocumentProcessor()
        
        # Update file path and metadata in document
        document.file_path = file_path
        document.doc_metadata = metadata
        db.commit()
        
        # Step 2: Extract text and create chunks
        document.processing_progress = 40
        document.processing_message = "Creating document chunks..."
        db.commit()
        
        # Process the document content
        from utils.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Use the existing process_document method but with the pre-created document
        try:
            # Extract text from the document based on file type
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif extension == ".md":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif extension == ".pdf":
                text = processor._extract_text_from_pdf(file_path)
            elif extension in [".docx", ".doc"]:
                text = processor._extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
            
            # Step 3: Split into chunks
            document.processing_progress = 50
            document.processing_message = "Splitting document into chunks..."
            db.commit()
            
            chunks = processor._split_text_into_chunks(text, chunk_size=1500, overlap=300)
            
            # Step 4: Create embeddings and store chunks
            document.processing_progress = 60
            document.processing_message = "Creating embeddings..."
            db.commit()
            
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                # Create embedding
                if processor.embeddings:
                    try:
                        embedding = processor.embeddings.embed_query(chunk)
                    except Exception as e:
                        print(f"Error creating embedding for chunk {i}: {e}")
                        embedding = processor._create_simple_embedding(chunk)
                else:
                    embedding = processor._create_simple_embedding(chunk)
                
                # Create chunk record
                chunk_metadata = {
                    "index": i,
                    "document_id": document.id,
                    "document_title": title,
                    "document_version": version,
                    "section": f"chunk_{i}",
                }
                
                chunk_obj = Chunk(
                    content=chunk,
                    embedding=processor._serialize_embedding(embedding),
                    chunk_metadata=chunk_metadata
                )
                
                db.add(chunk_obj)
                chunk_objects.append(chunk_obj)
            
            db.commit()
            
            # Associate chunks with document
            for chunk_obj in chunk_objects:
                document.chunks.append(chunk_obj)
            
            db.commit()
            
            # Step 5: Create index cache
            document.processing_progress = 70
            document.processing_message = "Creating document index..."
            db.commit()
            
            processor._create_simple_index(document.id, chunks)
            
            # Step 6: Generate learning content
            document.processing_progress = 80
            document.processing_message = "Generating learning content and quizzes..."
            db.commit()
            
            from utils.content_generator import ContentGenerator
            content_generator = ContentGenerator(db)
            success, message = content_generator.generate_learning_content(document.id)
            
            if not success:
                print(f"Warning: Content generation failed: {message}")
                document.processing_message = f"Processing completed with content generation warning: {message}"
            else:
                document.processing_message = "Document processing completed successfully!"
            
            # Step 7: Complete processing
            document.processing_progress = 100
            document.processing_status = "completed"
            document.processing_completed_at = datetime.utcnow()
            db.commit()
            
            print(f"Successfully completed background processing for document {document_id}")
            
        except Exception as process_error:
            print(f"Error during document processing: {process_error}")
            traceback.print_exc()
            
            document.processing_status = "failed"
            document.processing_message = f"Processing failed: {str(process_error)}"
            document.processing_completed_at = datetime.utcnow()
            db.commit()
            
    except Exception as e:
        print(f"Error in background processing for document {document_id}: {e}")
        traceback.print_exc()
        
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = "failed"
                document.processing_message = f"Background processing failed: {str(e)}"
                document.processing_completed_at = datetime.utcnow()
                db.commit()
        except:
            pass  # Don't let database errors crash the background task
            
    finally:
        db.close()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: str = Form(...),
    version: str = Form(...),
    effective_date: str = Form(...),
    language: str = Form("en"),
    db: Session = Depends(get_db)
):
    """Upload a document and start background processing."""
    try:
        # Validate file and filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate safe filename with unique identifier
        safe_filename = f"{uuid.uuid4()}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Ensure we're not overwriting a directory
        if os.path.isdir(file_path):
            raise HTTPException(status_code=400, detail="Invalid filename - conflicts with directory")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse the effective date
        effective_date_obj = datetime.strptime(effective_date, "%Y-%m-%d")
        
        # Create document record immediately with pending status
        metadata = {
            "filename": file.filename, 
            "original_filename": file.filename,
            "safe_filename": safe_filename,
            "file_size": file.size,
            "file_extension": file_extension
        }
        
        document = Document(
            title=title,
            document_type=document_type,
            version=version,
            effective_date=effective_date_obj,
            language=language,
            file_path=file_path,
            doc_metadata=metadata,
            processing_status="pending",
            processing_progress=0,
            processing_message="Document uploaded, processing will start shortly..."
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document.id,
            file_path,
            title,
            document_type,
            version,
            effective_date_obj,
            language,
            metadata
        )
        
        # Return immediate response
        return DocumentResponse(
            id=document.id,
            title=document.title,
            document_type=document.document_type,
            version=document.version,
            effective_date=document.effective_date,
            language=document.language,
            created_at=document.created_at,
            processing_status=document.processing_status,
            processing_progress=document.processing_progress,
            processing_message=document.processing_message
        )
        
    except Exception as e:
        # Clean up the file if processing fails
        if 'file_path' in locals() and os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up file {file_path}: {cleanup_error}")
        
        # Log the full error for debugging
        print(f"Document upload error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/status/{document_id}")
def get_document_processing_status(document_id: int, db: Session = Depends(get_db)):
    """Get the processing status of a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document.id,
        "title": document.title,
        "processing_status": document.processing_status,
        "processing_progress": document.processing_progress,
        "processing_message": document.processing_message,
        "processing_started_at": document.processing_started_at,
        "processing_completed_at": document.processing_completed_at,
        "created_at": document.created_at
    }

@router.get("/", response_model=List[DocumentResponse])
def get_documents(
    document_type: Optional[str] = None,
    language: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all documents with optional filters."""
    query = db.query(Document)
    
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    if language:
        query = query.filter(Document.language == language)
    
    documents = query.all()
    
    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            document_type=doc.document_type,
            version=doc.version,
            effective_date=doc.effective_date,
            language=doc.language,
            created_at=doc.created_at,
            processing_status=doc.processing_status or "completed",  # Legacy documents default to completed
            processing_progress=doc.processing_progress or 100,
            processing_message=doc.processing_message,
            processing_started_at=doc.processing_started_at,
            processing_completed_at=doc.processing_completed_at
        )
        for doc in documents
    ]

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get a specific document by ID."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        document_type=document.document_type,
        version=document.version,
        effective_date=document.effective_date,
        language=document.language,
        created_at=document.created_at,
        processing_status=document.processing_status or "completed",  # Legacy documents default to completed
        processing_progress=document.processing_progress or 100,
        processing_message=document.processing_message,
        processing_started_at=document.processing_started_at,
        processing_completed_at=document.processing_completed_at
    )

@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document by ID."""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete the file from storage
        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/ai-status")
def get_ai_status(db: Session = Depends(get_db)):
    """Get the current AI model status."""
    try:
        rag_engine = RAGEngine(db)
        return {
            "ai_model": rag_engine.get_ai_model_status(),
            "openai_available": rag_engine.openai_api_key is not None,
            "gemini_available": rag_engine.gemini_api_key is not None,
            "openrouter_available": rag_engine.openrouter_api_key is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting AI status: {str(e)}")

@router.get("/debug/database")
def get_database_debug_info(db: Session = Depends(get_db)):
    """Get database debug information."""
    try:
        rag_engine = RAGEngine(db)
        return rag_engine.get_database_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database debug info: {str(e)}")

@router.post("/debug/reprocess/{document_id}")
def reprocess_document(document_id: int, db: Session = Depends(get_db)):
    """Reprocess a document to recreate chunks."""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete existing chunks for this document
        existing_chunks = db.query(Chunk).filter(
            Chunk.chunk_metadata.op('->>')('document_id') == str(document_id)
        ).all()
        for chunk in existing_chunks:
            db.delete(chunk)
        db.commit()
        
        # Reprocess the document
        processor = DocumentProcessor()
        processor.process_document(
            file_path=document.file_path,
            title=document.title,
            document_type=document.document_type,
            version=document.version,
            effective_date=document.effective_date,
            language=document.language,
            metadata=document.doc_metadata
        )
        
        return {"message": f"Document {document_id} reprocessed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")

@router.post("/query")
def query_documents(query: DocumentQuery, db: Session = Depends(get_db)):
    """Query documents using RAG."""
    try:
        rag_engine = RAGEngine(db)
        result = rag_engine.query(
            query=query.query,
            document_ids=query.document_ids,
            procedure_mode=query.procedure_mode,
            language=query.language
        )
        
        return {
            "query": query.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "procedure_mode": query.procedure_mode,
            "language": query.language,
            "ai_model": rag_engine.get_ai_model_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")
