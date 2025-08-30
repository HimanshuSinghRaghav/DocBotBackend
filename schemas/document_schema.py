from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    document_type: str
    version: str
    language: str = "en"

class DocumentCreate(DocumentBase):
    effective_date: datetime

class DocumentResponse(DocumentBase):
    id: int
    effective_date: datetime
    created_at: datetime
    
    # Processing status fields
    processing_status: str = "pending"
    processing_progress: int = 0
    processing_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DocumentQuery(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    procedure_mode: bool = False
    language: str = "en"
