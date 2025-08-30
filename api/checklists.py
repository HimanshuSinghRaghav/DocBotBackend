from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from database.database import get_db
from models.models import Checklist, ChecklistCompletion, Document, User
from schemas.checklist_schema import (
    ChecklistCreate, ChecklistResponse, ChecklistCompletionCreate, ChecklistCompletionResponse
)

router = APIRouter()

@router.post("/", response_model=ChecklistResponse)
def create_checklist(checklist: ChecklistCreate, db: Session = Depends(get_db)):
    """Create a new checklist."""
    # Check if document exists
    if checklist.document_id:
        document = db.query(Document).filter(Document.id == checklist.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
    
    # Create checklist
    db_checklist = Checklist(
        title=checklist.title,
        description=checklist.description,
        frequency=checklist.frequency,
        items=checklist.items,
        document_id=checklist.document_id
    )
    db.add(db_checklist)
    db.commit()
    db.refresh(db_checklist)
    
    return ChecklistResponse(
        id=db_checklist.id,
        title=db_checklist.title,
        description=db_checklist.description,
        frequency=db_checklist.frequency,
        items=db_checklist.items,
        document_id=db_checklist.document_id,
        created_at=db_checklist.created_at
    )

@router.get("/", response_model=List[ChecklistResponse])
def get_checklists(
    frequency: Optional[str] = None,
    document_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all checklists with optional filters."""
    query = db.query(Checklist)
    
    if frequency:
        query = query.filter(Checklist.frequency == frequency)
    
    if document_id:
        query = query.filter(Checklist.document_id == document_id)
    
    checklists = query.all()
    
    return [
        ChecklistResponse(
            id=checklist.id,
            title=checklist.title,
            description=checklist.description,
            frequency=checklist.frequency,
            items=checklist.items,
            document_id=checklist.document_id,
            created_at=checklist.created_at
        )
        for checklist in checklists
    ]

@router.get("/{checklist_id}", response_model=ChecklistResponse)
def get_checklist(checklist_id: int, db: Session = Depends(get_db)):
    """Get a specific checklist by ID."""
    checklist = db.query(Checklist).filter(Checklist.id == checklist_id).first()
    
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    return ChecklistResponse(
        id=checklist.id,
        title=checklist.title,
        description=checklist.description,
        frequency=checklist.frequency,
        items=checklist.items,
        document_id=checklist.document_id,
        created_at=checklist.created_at
    )

@router.post("/completions", response_model=ChecklistCompletionResponse)
def create_checklist_completion(
    completion: ChecklistCompletionCreate,
    db: Session = Depends(get_db)
):
    """Record a checklist completion."""
    # Check if checklist exists
    checklist = db.query(Checklist).filter(Checklist.id == completion.checklist_id).first()
    if not checklist:
        raise HTTPException(status_code=404, detail="Checklist not found")
    
    # Check if user exists
    user = db.query(User).filter(User.id == completion.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create completion record
    db_completion = ChecklistCompletion(
        checklist_id=completion.checklist_id,
        user_id=completion.user_id,
        responses=completion.responses,
        attestation=completion.attestation
    )
    db.add(db_completion)
    db.commit()
    db.refresh(db_completion)
    
    return ChecklistCompletionResponse(
        id=db_completion.id,
        checklist_id=db_completion.checklist_id,
        user_id=db_completion.user_id,
        responses=db_completion.responses,
        attestation=db_completion.attestation,
        completed_at=db_completion.completed_at
    )

@router.get("/completions", response_model=List[ChecklistCompletionResponse])
def get_checklist_completions(
    checklist_id: Optional[int] = None,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get checklist completions with optional filters."""
    query = db.query(ChecklistCompletion)
    
    if checklist_id:
        query = query.filter(ChecklistCompletion.checklist_id == checklist_id)
    
    if user_id:
        query = query.filter(ChecklistCompletion.user_id == user_id)
    
    completions = query.all()
    
    return [
        ChecklistCompletionResponse(
            id=completion.id,
            checklist_id=completion.checklist_id,
            user_id=completion.user_id,
            responses=completion.responses,
            attestation=completion.attestation,
            completed_at=completion.completed_at
        )
        for completion in completions
    ]
