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

    class Config:
        from_attributes = True

class DocumentQuery(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    procedure_mode: bool = False
    language: str = "en"
