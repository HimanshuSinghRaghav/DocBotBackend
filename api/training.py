from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from database.database import get_db
from models.models import Quiz, Question, QuizAttempt, TrainingRecord, Document, User
from schemas.training_schema import (
    QuizCreate, QuizResponse, QuestionCreate, QuestionResponse,
    QuizAttemptCreate, QuizAttemptResponse, TrainingRecordCreate, TrainingRecordResponse
)

router = APIRouter()

@router.post("/quizzes", response_model=QuizResponse)
def create_quiz(quiz: QuizCreate, db: Session = Depends(get_db)):
    """Create a new quiz."""
    # Check if document exists
    document = db.query(Document).filter(Document.id == quiz.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Create quiz
    db_quiz = Quiz(
        title=quiz.title,
        document_id=quiz.document_id
    )
    db.add(db_quiz)
    db.commit()
    db.refresh(db_quiz)
    
    return QuizResponse(
        id=db_quiz.id,
        title=db_quiz.title,
        document_id=db_quiz.document_id,
        created_at=db_quiz.created_at
    )

@router.get("/quizzes", response_model=List[QuizResponse])
def get_quizzes(
    document_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all quizzes with optional document filter."""
    query = db.query(Quiz)
    
    if document_id:
        query = query.filter(Quiz.document_id == document_id)
    
    quizzes = query.all()
    
    return [
        QuizResponse(
            id=quiz.id,
            title=quiz.title,
            document_id=quiz.document_id,
            created_at=quiz.created_at
        )
        for quiz in quizzes
    ]

@router.get("/quizzes/{quiz_id}", response_model=QuizResponse)
def get_quiz(quiz_id: int, db: Session = Depends(get_db)):
    """Get a specific quiz by ID."""
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    return QuizResponse(
        id=quiz.id,
        title=quiz.title,
        document_id=quiz.document_id,
        created_at=quiz.created_at
    )

@router.post("/quizzes/{quiz_id}/questions", response_model=QuestionResponse)
def create_question(
    quiz_id: int,
    question: QuestionCreate,
    db: Session = Depends(get_db)
):
    """Create a new question for a quiz."""
    # Check if quiz exists
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Create question
    db_question = Question(
        quiz_id=quiz_id,
        question_text=question.question_text,
        question_type=question.question_type,
        options=question.options,
        correct_answer=question.correct_answer,
        explanation=question.explanation,
        source_chunk_id=question.source_chunk_id
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    return QuestionResponse(
        id=db_question.id,
        quiz_id=db_question.quiz_id,
        question_text=db_question.question_text,
        question_type=db_question.question_type,
        options=db_question.options,
        correct_answer=db_question.correct_answer,
        explanation=db_question.explanation,
        source_chunk_id=db_question.source_chunk_id
    )

@router.get("/quizzes/{quiz_id}/questions", response_model=List[QuestionResponse])
def get_questions(quiz_id: int, db: Session = Depends(get_db)):
    """Get all questions for a specific quiz."""
    # Check if quiz exists
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    questions = db.query(Question).filter(Question.quiz_id == quiz_id).all()
    
    return [
        QuestionResponse(
            id=question.id,
            quiz_id=question.quiz_id,
            question_text=question.question_text,
            question_type=question.question_type,
            options=question.options,
            correct_answer=question.correct_answer,
            explanation=question.explanation,
            source_chunk_id=question.source_chunk_id
        )
        for question in questions
    ]

@router.post("/attempts", response_model=QuizAttemptResponse)
def create_quiz_attempt(
    attempt: QuizAttemptCreate,
    db: Session = Depends(get_db)
):
    """Record a quiz attempt."""
    # Check if user exists
    user = db.query(User).filter(User.id == attempt.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if quiz exists
    quiz = db.query(Quiz).filter(Quiz.id == attempt.quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    # Create attempt
    db_attempt = QuizAttempt(
        user_id=attempt.user_id,
        quiz_id=attempt.quiz_id,
        score=attempt.score,
        answers=attempt.answers
    )
    db.add(db_attempt)
    db.commit()
    db.refresh(db_attempt)
    
    return QuizAttemptResponse(
        id=db_attempt.id,
        user_id=db_attempt.user_id,
        quiz_id=db_attempt.quiz_id,
        score=db_attempt.score,
        answers=db_attempt.answers,
        completed_at=db_attempt.completed_at
    )

@router.post("/records", response_model=TrainingRecordResponse)
def create_training_record(
    record: TrainingRecordCreate,
    db: Session = Depends(get_db)
):
    """Create or update a training record."""
    # Check if user exists
    user = db.query(User).filter(User.id == record.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if document exists
    document = db.query(Document).filter(Document.id == record.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if record already exists
    existing_record = db.query(TrainingRecord).filter(
        TrainingRecord.user_id == record.user_id,
        TrainingRecord.document_id == record.document_id
    ).first()
    
    if existing_record:
        # Update existing record
        existing_record.status = record.status
        existing_record.progress = record.progress
        existing_record.last_accessed = datetime.now()
        
        if record.status == "Completed" and existing_record.status != "Completed":
            existing_record.completed_at = datetime.now()
        
        db.commit()
        db.refresh(existing_record)
        
        return TrainingRecordResponse(
            id=existing_record.id,
            user_id=existing_record.user_id,
            document_id=existing_record.document_id,
            status=existing_record.status,
            progress=existing_record.progress,
            last_accessed=existing_record.last_accessed,
            completed_at=existing_record.completed_at
        )
    else:
        # Create new record
        db_record = TrainingRecord(
            user_id=record.user_id,
            document_id=record.document_id,
            status=record.status,
            progress=record.progress,
            last_accessed=datetime.now()
        )
        
        if record.status == "Completed":
            db_record.completed_at = datetime.now()
        
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return TrainingRecordResponse(
            id=db_record.id,
            user_id=db_record.user_id,
            document_id=db_record.document_id,
            status=db_record.status,
            progress=db_record.progress,
            last_accessed=db_record.last_accessed,
            completed_at=db_record.completed_at
        )

@router.get("/records", response_model=List[TrainingRecordResponse])
def get_training_records(
    user_id: Optional[int] = None,
    document_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get training records with optional filters."""
    query = db.query(TrainingRecord)
    
    if user_id:
        query = query.filter(TrainingRecord.user_id == user_id)
    
    if document_id:
        query = query.filter(TrainingRecord.document_id == document_id)
    
    if status:
        query = query.filter(TrainingRecord.status == status)
    
    records = query.all()
    
    return [
        TrainingRecordResponse(
            id=record.id,
            user_id=record.user_id,
            document_id=record.document_id,
            status=record.status,
            progress=record.progress,
            last_accessed=record.last_accessed,
            completed_at=record.completed_at
        )
        for record in records
    ]
