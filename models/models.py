from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.database import Base

# Association tables
document_chunk = Table(
    'document_chunk',
    Base.metadata,
    Column('document_id', Integer, ForeignKey('documents.id')),
    Column('chunk_id', Integer, ForeignKey('chunks.id'))
)

# Main models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)  # userid
    role = Column(String)  # admin, shift_lead, crew
    password = Column(String)  # hashed password
    location = Column(String, nullable=True)  # Keep for backward compatibility, make optional
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    training_records = relationship("TrainingRecord", back_populates="user")
    quiz_attempts = relationship("QuizAttempt", back_populates="user")
    checklist_completions = relationship("ChecklistCompletion", back_populates="user")
    module_progress = relationship("ModuleProgress", back_populates="user")
    session_progress = relationship("SessionProgress", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    document_type = Column(String)  # SOP, Training, Checklist
    version = Column(String)
    effective_date = Column(DateTime)
    language = Column(String, default="en")
    file_path = Column(String)
    doc_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    chunks = relationship("Chunk", secondary=document_chunk, back_populates="documents")
    quizzes = relationship("Quiz", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text)
    embedding = Column(Text)  # Store as serialized vector
    chunk_metadata = Column(JSON)  # section, page, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    documents = relationship("Document", secondary=document_chunk, back_populates="chunks")

class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    document_id = Column(Integer, ForeignKey("documents.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="quizzes")
    questions = relationship("Question", back_populates="quiz")
    quiz_attempts = relationship("QuizAttempt", back_populates="quiz")

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    question_text = Column(Text)
    question_type = Column(String)  # MCQ, True/False
    options = Column(JSON)
    correct_answer = Column(String)
    explanation = Column(Text)
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    quiz = relationship("Quiz", back_populates="questions")
    source_chunk = relationship("Chunk")

class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    score = Column(Float)
    answers = Column(JSON)  # Store user's answers
    completed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="quiz_attempts")
    quiz = relationship("Quiz", back_populates="quiz_attempts")

class TrainingRecord(Base):
    __tablename__ = "training_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    status = Column(String)  # Not Started, In Progress, Completed
    progress = Column(Float, default=0)  # 0-100%
    last_accessed = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="training_records")
    document = relationship("Document")

class Checklist(Base):
    __tablename__ = "checklists"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    frequency = Column(String)  # Daily, Weekly, Monthly
    items = Column(JSON)  # List of checklist items
    document_id = Column(Integer, ForeignKey("documents.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    document = relationship("Document")
    completions = relationship("ChecklistCompletion", back_populates="checklist")

class ChecklistCompletion(Base):
    __tablename__ = "checklist_completions"

    id = Column(Integer, primary_key=True, index=True)
    checklist_id = Column(Integer, ForeignKey("checklists.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    responses = Column(JSON)  # User's responses to checklist items
    attestation = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    checklist = relationship("Checklist", back_populates="completions")
    user = relationship("User", back_populates="checklist_completions")

class LearningModule(Base):
    __tablename__ = "learning_modules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    document_id = Column(Integer, ForeignKey("documents.id"))
    module_order = Column(Integer, default=1)  # Order within the document
    estimated_duration = Column(Integer)  # In minutes
    difficulty_level = Column(String, default="Beginner")  # Beginner, Intermediate, Advanced
    learning_objectives = Column(JSON)  # List of learning objectives
    prerequisites = Column(JSON)  # List of prerequisite modules
    module_metadata = Column(JSON)  # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    document = relationship("Document")
    learning_sessions = relationship("LearningSession", back_populates="module", cascade="all, delete-orphan")
    module_progress = relationship("ModuleProgress", back_populates="module")

class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    module_id = Column(Integer, ForeignKey("learning_modules.id"))
    session_order = Column(Integer, default=1)  # Order within the module
    session_type = Column(String, default="mixed")  # quiz, lesson, mixed, assessment
    estimated_duration = Column(Integer)  # In minutes
    passing_score = Column(Integer, default=70)  # Minimum score to pass (for quiz sessions)
    max_attempts = Column(Integer, default=3)  # Maximum attempts allowed
    is_required = Column(Boolean, default=True)  # Whether this session is required
    unlock_conditions = Column(JSON)  # Conditions to unlock this session
    session_metadata = Column(JSON)  # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    module = relationship("LearningModule", back_populates="learning_sessions")
    session_content = relationship("SessionContent", back_populates="session", cascade="all, delete-orphan")
    session_progress = relationship("SessionProgress", back_populates="session")

class SessionContent(Base):
    __tablename__ = "session_content"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("learning_sessions.id"))
    content_type = Column(String)  # lesson, quiz_question, instruction
    content_order = Column(Integer, default=1)  # Order within the session
    
    # For lessons
    lesson_data = Column(JSON)  # Lesson content (title, summary, key_points, etc.)
    
    # For quiz questions
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=True)
    
    # For instructions/text content
    instruction_text = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("LearningSession", back_populates="session_content")
    question = relationship("Question")

class ModuleProgress(Base):
    __tablename__ = "module_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    module_id = Column(Integer, ForeignKey("learning_modules.id"))
    status = Column(String, default="not_started")  # not_started, in_progress, completed, failed
    progress_percentage = Column(Float, default=0.0)  # 0-100
    current_session_id = Column(Integer, ForeignKey("learning_sessions.id"), nullable=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User")
    module = relationship("LearningModule", back_populates="module_progress")
    current_session = relationship("LearningSession")

class SessionProgress(Base):
    __tablename__ = "session_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("learning_sessions.id"))
    status = Column(String, default="not_started")  # not_started, in_progress, completed, failed
    score = Column(Float, nullable=True)  # Quiz score if applicable
    attempts = Column(Integer, default=0)  # Number of attempts
    time_spent = Column(Integer, default=0)  # Time spent in seconds
    answers = Column(JSON)  # User's answers for quiz sessions
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User")
    session = relationship("LearningSession", back_populates="session_progress")
