from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional, Dict, Any
from datetime import datetime

from database.database import get_db
from models.models import (
    LearningModule, LearningSession, SessionContent, Question, 
    ModuleProgress, SessionProgress, Document, User
)
from schemas.learning_schema import (
    LearningModuleCreate, LearningModuleResponse, LearningModuleWithSessions,
    LearningSessionCreate, LearningSessionResponse, LearningSessionWithContent,
    SessionContentCreate, SessionContentResponse,
    ModuleProgressResponse, SessionProgressResponse,
    LearningPathResponse, SessionAttemptRequest, SessionAttemptResponse
)

router = APIRouter()

# Learning Modules
@router.post("/modules", response_model=LearningModuleResponse)
def create_learning_module(
    module: LearningModuleCreate,
    db: Session = Depends(get_db)
):
    """Create a new learning module."""
    # Check if document exists
    document = db.query(Document).filter(Document.id == module.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    db_module = LearningModule(
        title=module.title,
        description=module.description,
        document_id=module.document_id,
        module_order=module.module_order,
        estimated_duration=module.estimated_duration,
        difficulty_level=module.difficulty_level,
        learning_objectives=module.learning_objectives,
        prerequisites=module.prerequisites
    )
    
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    
    return db_module

@router.get("/modules", response_model=List[LearningModuleResponse])
def get_learning_modules(
    document_id: Optional[int] = None,
    difficulty_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get learning modules with optional filters."""
    query = db.query(LearningModule)
    
    if document_id:
        query = query.filter(LearningModule.document_id == document_id)
    
    if difficulty_level:
        query = query.filter(LearningModule.difficulty_level == difficulty_level)
    
    modules = query.order_by(LearningModule.module_order).all()
    
    # Add session count to each module
    for module in modules:
        module.sessions_count = len(module.learning_sessions)
    
    return modules

@router.get("/modules/{module_id}", response_model=LearningModuleWithSessions)
def get_learning_module(
    module_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get a specific learning module with its sessions."""
    module = db.query(LearningModule).filter(LearningModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Learning module not found")
    
    # Get sessions for this module
    sessions = db.query(LearningSession).filter(
        LearningSession.module_id == module_id
    ).order_by(LearningSession.session_order).all()
    
    # If user_id provided, check unlock status and progress for each session
    if user_id:
        for session in sessions:
            # Check if session is unlocked
            session.is_unlocked = _is_session_unlocked(session, user_id, db)
            
            # Get user progress for this session
            progress = db.query(SessionProgress).filter(
                and_(
                    SessionProgress.user_id == user_id,
                    SessionProgress.session_id == session.id
                )
            ).first()
            
            if progress:
                session.user_progress = {
                    "status": progress.status,
                    "score": progress.score,
                    "attempts": progress.attempts,
                    "last_accessed": progress.last_accessed
                }
            else:
                session.user_progress = {
                    "status": "not_started",
                    "score": None,
                    "attempts": 0,
                    "last_accessed": None
                }
            
            # Add content count
            session.content_count = len(session.session_content)
    
    module.learning_sessions = sessions
    return module

# Learning Sessions
@router.post("/sessions", response_model=LearningSessionResponse)
def create_learning_session(
    session: LearningSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new learning session."""
    # Check if module exists
    module = db.query(LearningModule).filter(LearningModule.id == session.module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Learning module not found")
    
    db_session = LearningSession(
        title=session.title,
        description=session.description,
        module_id=session.module_id,
        session_order=session.session_order,
        session_type=session.session_type,
        estimated_duration=session.estimated_duration,
        passing_score=session.passing_score,
        max_attempts=session.max_attempts,
        is_required=session.is_required,
        unlock_conditions=session.unlock_conditions
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return db_session

@router.get("/sessions/{session_id}", response_model=LearningSessionWithContent)
def get_learning_session(
    session_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get a specific learning session with its content."""
    session = db.query(LearningSession).filter(LearningSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Learning session not found")
    
    # Check if user has access to this session
    if user_id and not _is_session_unlocked(session, user_id, db):
        raise HTTPException(status_code=403, detail="Session is locked")
    
    # Get session content with questions
    content = db.query(SessionContent).filter(
        SessionContent.session_id == session_id
    ).order_by(SessionContent.content_order).all()
    
    # Load question details for quiz content
    for item in content:
        if item.question_id:
            question = db.query(Question).filter(Question.id == item.question_id).first()
            if question:
                item.question = {
                    "id": question.id,
                    "question_text": question.question_text,
                    "question_type": question.question_type,
                    "options": question.options,
                    "explanation": question.explanation
                    # Note: Don't include correct_answer for security
                }
    
    session.session_content = content
    return session

# Session Content Management
@router.post("/sessions/{session_id}/content", response_model=SessionContentResponse)
def add_session_content(
    session_id: int,
    content: SessionContentCreate,
    db: Session = Depends(get_db)
):
    """Add content to a learning session."""
    # Verify session exists
    session = db.query(LearningSession).filter(LearningSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Learning session not found")
    
    # Validate content based on type
    if content.content_type == "quiz_question" and not content.question_id:
        raise HTTPException(status_code=400, detail="Question ID required for quiz content")
    elif content.content_type == "lesson" and not content.lesson_data:
        raise HTTPException(status_code=400, detail="Lesson data required for lesson content")
    elif content.content_type == "instruction" and not content.instruction_text:
        raise HTTPException(status_code=400, detail="Instruction text required for instruction content")
    
    db_content = SessionContent(
        session_id=session_id,
        content_type=content.content_type,
        content_order=content.content_order,
        lesson_data=content.lesson_data,
        question_id=content.question_id,
        instruction_text=content.instruction_text
    )
    
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    
    return db_content

# User Progress and Attempts
@router.post("/sessions/{session_id}/attempt", response_model=SessionAttemptResponse)
def attempt_session(
    session_id: int,
    attempt: SessionAttemptRequest,
    user_id: int = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Submit an attempt for a learning session."""
    session = db.query(LearningSession).filter(LearningSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Learning session not found")
    
    # Check if session is unlocked
    if not _is_session_unlocked(session, user_id, db):
        raise HTTPException(status_code=403, detail="Session is locked")
    
    # Get or create session progress
    progress = db.query(SessionProgress).filter(
        and_(
            SessionProgress.user_id == user_id,
            SessionProgress.session_id == session_id
        )
    ).first()
    
    if not progress:
        progress = SessionProgress(
            user_id=user_id,
            session_id=session_id,
            started_at=datetime.utcnow()
        )
        db.add(progress)
    
    # Update attempt count
    progress.attempts += 1
    progress.last_accessed = datetime.utcnow()
    progress.time_spent += attempt.time_spent or 0
    
    # Check if max attempts exceeded
    if progress.attempts > session.max_attempts:
        raise HTTPException(status_code=400, detail="Maximum attempts exceeded")
    
    # Calculate score for quiz sessions
    score = None
    passed = True
    
    if session.session_type in ["quiz", "assessment"] and attempt.answers:
        score = _calculate_session_score(session_id, attempt.answers, db)
        passed = score >= session.passing_score
        progress.score = score
        progress.answers = attempt.answers
    
    # Update progress status
    if passed:
        progress.status = "completed"
        progress.completed_at = datetime.utcnow()
        
        # Update module progress
        _update_module_progress(session.module_id, user_id, db)
    else:
        progress.status = "failed" if progress.attempts >= session.max_attempts else "in_progress"
    
    db.commit()
    
    # Find next session
    next_session = _get_next_unlocked_session(session, user_id, db)
    
    # Check if module is completed
    module_completed = _is_module_completed(session.module_id, user_id, db)
    
    return SessionAttemptResponse(
        session_id=session_id,
        score=score,
        passed=passed,
        feedback=_generate_session_feedback(score, passed, session.passing_score),
        next_session_id=next_session.id if next_session else None,
        module_completed=module_completed
    )

@router.get("/modules/{module_id}/progress")
def get_module_progress(
    module_id: int,
    user_id: int = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Get user's progress for a specific module."""
    module = db.query(LearningModule).filter(LearningModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Learning module not found")
    
    progress = db.query(ModuleProgress).filter(
        and_(
            ModuleProgress.user_id == user_id,
            ModuleProgress.module_id == module_id
        )
    ).first()
    
    if not progress:
        # Create initial progress record
        progress = ModuleProgress(
            user_id=user_id,
            module_id=module_id,
            status="not_started"
        )
        db.add(progress)
        db.commit()
        db.refresh(progress)
    
    return progress

@router.get("/documents/{document_id}/learning-path", response_model=LearningPathResponse)
def get_learning_path(
    document_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get the complete learning path for a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get all modules for this document
    modules = db.query(LearningModule).filter(
        LearningModule.document_id == document_id
    ).order_by(LearningModule.module_order).all()
    
    total_sessions = 0
    completed_sessions = 0
    completed_modules = 0
    total_duration = 0
    
    # Process each module
    for module in modules:
        sessions = db.query(LearningSession).filter(
            LearningSession.module_id == module.id
        ).all()
        
        module.learning_sessions = sessions
        total_sessions += len(sessions)
        
        if module.estimated_duration:
            total_duration += module.estimated_duration
        
        if user_id:
            # Check module completion
            module_progress = db.query(ModuleProgress).filter(
                and_(
                    ModuleProgress.user_id == user_id,
                    ModuleProgress.module_id == module.id
                )
            ).first()
            
            if module_progress and module_progress.status == "completed":
                completed_modules += 1
            
            # Count completed sessions in this module
            for session in sessions:
                session_progress = db.query(SessionProgress).filter(
                    and_(
                        SessionProgress.user_id == user_id,
                        SessionProgress.session_id == session.id,
                        SessionProgress.status == "completed"
                    )
                ).first()
                
                if session_progress:
                    completed_sessions += 1
    
    overall_progress = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
    
    return LearningPathResponse(
        document_id=document_id,
        document_title=document.title,
        total_modules=len(modules),
        completed_modules=completed_modules,
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        overall_progress=overall_progress,
        estimated_duration=total_duration,
        modules=modules
    )

# Helper Functions
def _is_session_unlocked(session: LearningSession, user_id: int, db: Session) -> bool:
    """Check if a session is unlocked for a user."""
    # First session in module is always unlocked
    if session.session_order == 1:
        return True
    
    # Check unlock conditions
    if session.unlock_conditions:
        # Custom unlock logic can be implemented here
        # For now, just check if previous session is completed
        prev_session = db.query(LearningSession).filter(
            and_(
                LearningSession.module_id == session.module_id,
                LearningSession.session_order == session.session_order - 1
            )
        ).first()
        
        if prev_session:
            prev_progress = db.query(SessionProgress).filter(
                and_(
                    SessionProgress.user_id == user_id,
                    SessionProgress.session_id == prev_session.id,
                    SessionProgress.status == "completed"
                )
            ).first()
            
            return prev_progress is not None
    
    return True

def _calculate_session_score(session_id: int, answers: Dict[str, Any], db: Session) -> float:
    """Calculate score for a quiz session."""
    # Get all questions in this session
    content = db.query(SessionContent).filter(
        and_(
            SessionContent.session_id == session_id,
            SessionContent.content_type == "quiz_question"
        )
    ).all()
    
    if not content:
        return 100.0
    
    correct_answers = 0
    total_questions = len(content)
    
    for item in content:
        if item.question_id:
            question = db.query(Question).filter(Question.id == item.question_id).first()
            if question and str(item.question_id) in answers:
                user_answer = answers[str(item.question_id)]
                if user_answer == question.correct_answer:
                    correct_answers += 1
    
    return (correct_answers / total_questions) * 100 if total_questions > 0 else 0

def _update_module_progress(module_id: int, user_id: int, db: Session):
    """Update module progress based on session completions."""
    # Get or create module progress
    progress = db.query(ModuleProgress).filter(
        and_(
            ModuleProgress.user_id == user_id,
            ModuleProgress.module_id == module_id
        )
    ).first()
    
    if not progress:
        progress = ModuleProgress(
            user_id=user_id,
            module_id=module_id,
            started_at=datetime.utcnow()
        )
        db.add(progress)
    
    # Count completed sessions
    sessions = db.query(LearningSession).filter(
        LearningSession.module_id == module_id
    ).all()
    
    completed_count = 0
    for session in sessions:
        session_progress = db.query(SessionProgress).filter(
            and_(
                SessionProgress.user_id == user_id,
                SessionProgress.session_id == session.id,
                SessionProgress.status == "completed"
            )
        ).first()
        
        if session_progress:
            completed_count += 1
    
    total_sessions = len(sessions)
    progress.progress_percentage = (completed_count / total_sessions * 100) if total_sessions > 0 else 0
    
    if completed_count == total_sessions:
        progress.status = "completed"
        progress.completed_at = datetime.utcnow()
    elif completed_count > 0:
        progress.status = "in_progress"
    
    progress.last_accessed = datetime.utcnow()
    db.commit()

def _get_next_unlocked_session(current_session: LearningSession, user_id: int, db: Session) -> Optional[LearningSession]:
    """Get the next unlocked session for a user."""
    next_session = db.query(LearningSession).filter(
        and_(
            LearningSession.module_id == current_session.module_id,
            LearningSession.session_order == current_session.session_order + 1
        )
    ).first()
    
    if next_session and _is_session_unlocked(next_session, user_id, db):
        return next_session
    
    return None

def _is_module_completed(module_id: int, user_id: int, db: Session) -> bool:
    """Check if a module is completed by a user."""
    progress = db.query(ModuleProgress).filter(
        and_(
            ModuleProgress.user_id == user_id,
            ModuleProgress.module_id == module_id,
            ModuleProgress.status == "completed"
        )
    ).first()
    
    return progress is not None

def _generate_session_feedback(score: Optional[float], passed: bool, passing_score: int) -> Optional[str]:
    """Generate feedback message for session attempt."""
    if score is None:
        return "Session completed successfully!"
    
    if passed:
        if score >= 90:
            return f"Excellent work! You scored {score:.1f}% and passed with flying colors!"
        elif score >= 80:
            return f"Great job! You scored {score:.1f}% and passed the session."
        else:
            return f"Good work! You scored {score:.1f}% and passed the session."
    else:
        return f"You scored {score:.1f}%, which is below the passing score of {passing_score}%. Please review the material and try again."