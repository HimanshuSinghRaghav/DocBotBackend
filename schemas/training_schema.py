from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class QuizBase(BaseModel):
    title: str
    document_id: int

class QuizCreate(QuizBase):
    pass

class QuizResponse(QuizBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class QuestionBase(BaseModel):
    question_text: str
    question_type: str  # MCQ, True/False
    options: Dict[str, str]  # {"A": "Option text", "B": "Option text"}
    correct_answer: str
    explanation: str
    source_chunk_id: Optional[int] = None

class QuestionCreate(QuestionBase):
    pass

class QuestionResponse(QuestionBase):
    id: int
    quiz_id: int

    class Config:
        from_attributes = True

class QuizAttemptBase(BaseModel):
    user_id: int
    quiz_id: int
    score: float
    answers: Dict[str, str]  # {"question_id": "selected_answer"}

class QuizAttemptCreate(QuizAttemptBase):
    pass

class QuizAttemptResponse(QuizAttemptBase):
    id: int
    completed_at: datetime

    class Config:
        from_attributes = True

class TrainingRecordBase(BaseModel):
    user_id: int
    document_id: int
    status: str  # Not Started, In Progress, Completed
    progress: float  # 0-100%

class TrainingRecordCreate(TrainingRecordBase):
    pass

class TrainingRecordResponse(TrainingRecordBase):
    id: int
    last_accessed: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
