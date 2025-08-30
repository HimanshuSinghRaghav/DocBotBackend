from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChecklistBase(BaseModel):
    title: str
    description: str
    frequency: str  # Daily, Weekly, Monthly
    items: List[Dict[str, Any]]  # List of checklist items with their details
    document_id: Optional[int] = None

class ChecklistCreate(ChecklistBase):
    pass

class ChecklistResponse(ChecklistBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ChecklistCompletionBase(BaseModel):
    checklist_id: int
    user_id: int
    responses: Dict[str, Any]  # User's responses to checklist items
    attestation: bool

class ChecklistCompletionCreate(ChecklistCompletionBase):
    pass

class ChecklistCompletionResponse(ChecklistCompletionBase):
    id: int
    completed_at: datetime

    class Config:
        from_attributes = True
