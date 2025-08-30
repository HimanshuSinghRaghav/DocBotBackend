from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Learning Module Schemas
class LearningModuleBase(BaseModel):
    title: str
    description: Optional[str] = None
    document_id: int
    module_order: int = 1
    estimated_duration: Optional[int] = None
    difficulty_level: str = "Beginner"
    learning_objectives: Optional[List[str]] = []
    prerequisites: Optional[List[int]] = []

class LearningModuleCreate(LearningModuleBase):
    pass

class LearningModuleResponse(LearningModuleBase):
    id: int
    module_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    sessions_count: Optional[int] = 0
    
    class Config:
        from_attributes = True

# Learning Session Schemas
class LearningSessionBase(BaseModel):
    title: str
    description: Optional[str] = None
    module_id: int
    session_order: int = 1
    session_type: str = "mixed"
    estimated_duration: Optional[int] = None
    passing_score: int = 70
    max_attempts: int = 3
    is_required: bool = True

class LearningSessionCreate(LearningSessionBase):
    unlock_conditions: Optional[Dict[str, Any]] = None

class LearningSessionResponse(LearningSessionBase):
    id: int
    unlock_conditions: Optional[Dict[str, Any]] = None
    session_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    content_count: Optional[int] = 0
    is_unlocked: Optional[bool] = None
    user_progress: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

# Session Content Schemas
class SessionContentBase(BaseModel):
    session_id: int
    content_type: str
    content_order: int = 1

class SessionContentCreate(SessionContentBase):
    lesson_data: Optional[Dict[str, Any]] = None
    question_id: Optional[int] = None
    instruction_text: Optional[str] = None

class SessionContentResponse(SessionContentBase):
    id: int
    lesson_data: Optional[Dict[str, Any]] = None
    question_id: Optional[int] = None
    instruction_text: Optional[str] = None
    created_at: datetime
    question: Optional[Dict[str, Any]] = None  # Question details if applicable
    
    class Config:
        from_attributes = True

# Progress Schemas
class ModuleProgressBase(BaseModel):
    user_id: int
    module_id: int
    status: str = "not_started"
    progress_percentage: float = 0.0

class ModuleProgressResponse(ModuleProgressBase):
    id: int
    current_session_id: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class SessionProgressBase(BaseModel):
    user_id: int
    session_id: int
    status: str = "not_started"

class SessionProgressCreate(SessionProgressBase):
    answers: Optional[Dict[str, Any]] = None

class SessionProgressResponse(SessionProgressBase):
    id: int
    score: Optional[float] = None
    attempts: int = 0
    time_spent: int = 0
    answers: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Complete Module with Sessions
class LearningModuleWithSessions(LearningModuleResponse):
    learning_sessions: List[LearningSessionResponse] = []

# Complete Session with Content
class LearningSessionWithContent(LearningSessionResponse):
    session_content: List[SessionContentResponse] = []

# Learning Path Overview
class LearningPathResponse(BaseModel):
    document_id: int
    document_title: str
    total_modules: int
    completed_modules: int
    total_sessions: int
    completed_sessions: int
    overall_progress: float
    estimated_duration: int
    modules: List[LearningModuleWithSessions] = []
    
    class Config:
        from_attributes = True

# Session Attempt
class SessionAttemptRequest(BaseModel):
    session_id: int
    answers: Optional[Dict[str, Any]] = None
    time_spent: Optional[int] = 0

class SessionAttemptResponse(BaseModel):
    session_id: int
    score: Optional[float] = None
    passed: bool
    feedback: Optional[str] = None
    next_session_id: Optional[int] = None
    module_completed: bool = False