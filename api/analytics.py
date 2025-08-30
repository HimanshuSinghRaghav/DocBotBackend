from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from database.database import get_db
from models.models import (
    User, Document, TrainingRecord, QuizAttempt, 
    ChecklistCompletion, Quiz, Checklist
)

router = APIRouter()

@router.get("/training/completion")
def get_training_completion(
    document_id: Optional[int] = None,
    role: Optional[str] = None,
    location: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get training completion statistics."""
    # Base query for users
    user_query = db.query(User)
    
    if role:
        user_query = user_query.filter(User.role == role)
    
    if location:
        user_query = user_query.filter(User.location == location)
    
    users = user_query.all()
    user_ids = [user.id for user in users]
    
    # Get training records for these users
    records_query = db.query(TrainingRecord).filter(TrainingRecord.user_id.in_(user_ids))
    
    if document_id:
        records_query = records_query.filter(TrainingRecord.document_id == document_id)
    
    records = records_query.all()
    
    # Calculate completion statistics
    total_users = len(users)
    completed_count = len([r for r in records if r.status == "Completed"])
    in_progress_count = len([r for r in records if r.status == "In Progress"])
    not_started_count = total_users - completed_count - in_progress_count
    
    completion_percentage = (completed_count / total_users * 100) if total_users > 0 else 0
    
    return {
        "total_users": total_users,
        "completed_count": completed_count,
        "in_progress_count": in_progress_count,
        "not_started_count": not_started_count,
        "completion_percentage": completion_percentage
    }

@router.get("/training/scores")
def get_training_scores(
    quiz_id: Optional[int] = None,
    role: Optional[str] = None,
    location: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get quiz score statistics."""
    # Base query for quiz attempts
    attempts_query = db.query(QuizAttempt)
    
    if quiz_id:
        attempts_query = attempts_query.filter(QuizAttempt.quiz_id == quiz_id)
    
    # Filter by user role or location if specified
    if role or location:
        attempts_query = attempts_query.join(User, QuizAttempt.user_id == User.id)
        
        if role:
            attempts_query = attempts_query.filter(User.role == role)
        
        if location:
            attempts_query = attempts_query.filter(User.location == location)
    
    attempts = attempts_query.all()
    
    if not attempts:
        return {
            "average_score": 0,
            "highest_score": 0,
            "lowest_score": 0,
            "attempt_count": 0
        }
    
    # Calculate score statistics
    scores = [attempt.score for attempt in attempts]
    average_score = sum(scores) / len(scores)
    highest_score = max(scores)
    lowest_score = min(scores)
    
    return {
        "average_score": average_score,
        "highest_score": highest_score,
        "lowest_score": lowest_score,
        "attempt_count": len(attempts)
    }

@router.get("/checklists/adherence")
def get_checklist_adherence(
    checklist_id: Optional[int] = None,
    role: Optional[str] = None,
    location: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get checklist adherence statistics."""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Base query for checklist completions
    completions_query = db.query(ChecklistCompletion).filter(
        ChecklistCompletion.completed_at >= start_date,
        ChecklistCompletion.completed_at <= end_date
    )
    
    if checklist_id:
        completions_query = completions_query.filter(ChecklistCompletion.checklist_id == checklist_id)
    
    # Filter by user role or location if specified
    if role or location:
        completions_query = completions_query.join(User, ChecklistCompletion.user_id == User.id)
        
        if role:
            completions_query = completions_query.filter(User.role == role)
        
        if location:
            completions_query = completions_query.filter(User.location == location)
    
    completions = completions_query.all()
    
    # Get all checklists
    checklists_query = db.query(Checklist)
    if checklist_id:
        checklists_query = checklists_query.filter(Checklist.id == checklist_id)
    
    checklists = checklists_query.all()
    
    # Calculate adherence statistics
    adherence_data = {}
    for checklist in checklists:
        checklist_completions = [c for c in completions if c.checklist_id == checklist.id]
        
        # Calculate expected completions based on frequency
        expected_count = 0
        if checklist.frequency == "Daily":
            expected_count = days
        elif checklist.frequency == "Weekly":
            expected_count = days // 7
        elif checklist.frequency == "Monthly":
            expected_count = days // 30
        
        actual_count = len(checklist_completions)
        adherence_percentage = (actual_count / expected_count * 100) if expected_count > 0 else 0
        
        adherence_data[checklist.title] = {
            "expected_count": expected_count,
            "actual_count": actual_count,
            "adherence_percentage": adherence_percentage
        }
    
    return adherence_data

@router.get("/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get a summary of key metrics for the dashboard."""
    # Count total users by role
    user_counts = db.query(User.role, func.count(User.id)).group_by(User.role).all()
    user_count_by_role = {role: count for role, count in user_counts}
    total_users = sum(user_count_by_role.values())
    
    # Training completion stats
    completed_count = db.query(func.count(TrainingRecord.id)).filter(
        TrainingRecord.status == "Completed"
    ).scalar() or 0
    
    in_progress_count = db.query(func.count(TrainingRecord.id)).filter(
        TrainingRecord.status == "In Progress"
    ).scalar() or 0
    
    # Quiz performance
    avg_score = db.query(func.avg(QuizAttempt.score)).scalar() or 0
    
    # Recent checklist completions
    recent_date = datetime.now() - timedelta(days=7)
    recent_checklists = db.query(func.count(ChecklistCompletion.id)).filter(
        ChecklistCompletion.completed_at >= recent_date
    ).scalar() or 0
    
    # Document coverage
    document_count = db.query(func.count(Document.id)).scalar() or 0
    
    return {
        "user_statistics": {
            "total_users": total_users,
            "by_role": user_count_by_role
        },
        "training_statistics": {
            "completed_count": completed_count,
            "in_progress_count": in_progress_count,
            "completion_percentage": (completed_count / total_users * 100) if total_users > 0 else 0
        },
        "quiz_statistics": {
            "average_score": avg_score
        },
        "checklist_statistics": {
            "recent_completions": recent_checklists
        },
        "document_statistics": {
            "total_documents": document_count
        }
    }
