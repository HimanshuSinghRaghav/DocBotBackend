from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from datetime import datetime

from database.database import get_db
from models.models import Document, Chunk
from schemas.document_schema import DocumentCreate, DocumentResponse, DocumentQuery
from utils.document_processor import DocumentProcessor
from utils.rag_engine import RAGEngine

router = APIRouter()

# Directory to store uploaded documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: str = Form(...),
    version: str = Form(...),
    effective_date: str = Form(...),
    language: str = Form("en"),
    db: Session = Depends(get_db)
):
    """Upload and process a new document."""
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse the effective date
        effective_date_obj = datetime.strptime(effective_date, "%Y-%m-%d")
        
        # Process the document using DocumentProcessor
        processor = DocumentProcessor()
        metadata = {"filename": file.filename, "file_size": file.size}
        document = processor.process_document(
            file_path=file_path,
            title=title,
            document_type=document_type,
            version=version,
            effective_date=effective_date_obj,
            language=language,
            metadata=metadata
        )
        
        # Check if there were extraction errors
        if document.doc_metadata and "extraction_error" in document.doc_metadata:
            # Return success but with a warning
            return DocumentResponse(
                id=document.id,
                title=document.title,
                message=f"Document uploaded with extraction issues: {document.doc_metadata['extraction_error']}",
                success=True,
                document=document
            )
        
        return DocumentResponse(
            id=document.id,
            title=document.title,
            document_type=document.document_type,
            version=document.version,
            effective_date=document.effective_date,
            language=document.language,
            created_at=document.created_at
        )
    except Exception as e:
        # Clean up the file if processing fails
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

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
            created_at=doc.created_at
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
        created_at=document.created_at
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
