#!/usr/bin/env python3
"""
Debug script to test database and chunk retrieval
"""

import os
import sys
from sqlalchemy.orm import Session
from database.database import SessionLocal
from models.models import Document, Chunk
from utils.rag_engine import RAGEngine

def test_database():
    """Test database connectivity and data."""
    db = SessionLocal()
    
    try:
        print("=== Database Debug Test ===")
        
        # Check documents
        docs = db.query(Document).all()
        print(f"Total documents: {len(docs)}")
        
        for doc in docs:
            print(f"\nDocument ID: {doc.id}")
            print(f"Title: {doc.title}")
            print(f"File path: {doc.file_path}")
            
            # Check chunks using JSON query
            chunks_json = db.query(Chunk).filter(
                Chunk.chunk_metadata.op('->>')('document_id') == str(doc.id)
            ).all()
            print(f"Chunks (JSON query): {len(chunks_json)}")
            
            # Check chunks using relationship
            chunks_rel = doc.chunks
            print(f"Chunks (relationship): {len(chunks_rel)}")
            
            # Show chunk content
            for i, chunk in enumerate(chunks_json[:2]):  # Show first 2 chunks
                print(f"  Chunk {i}: {chunk.content[:100]}...")
                print(f"  Metadata: {chunk.chunk_metadata}")
        
        # Check all chunks
        all_chunks = db.query(Chunk).all()
        print(f"\nTotal chunks in database: {len(all_chunks)}")
        
        # Test RAG engine
        print("\n=== Testing RAG Engine ===")
        rag_engine = RAGEngine(db)
        
        # Test database status
        status = rag_engine.get_database_status()
        print(f"Database status: {status}")
        
        # Test query
        if docs:
            result = rag_engine.query("test query", document_ids=[docs[0].id])
            print(f"Query result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_database()
