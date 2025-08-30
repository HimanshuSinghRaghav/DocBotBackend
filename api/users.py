from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from database.database import get_db
from models.models import User
from schemas.user_schema import UserCreate, UserResponse, UserUpdate, UserResponseWithToken, UserLogin
from utils.auth import hash_password, create_access_token, get_current_user_from_token, verify_password

router = APIRouter()

@router.post("/", response_model=UserResponseWithToken)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with name, email, role, and password. Returns user data with JWT token."""
    # Validate role (additional server-side validation)
    allowed_roles = ["admin", "shift_lead", "crew"]
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid role. Must be one of: {', '.join(allowed_roles)}"
        )
    
    # Check if user with email already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = hash_password(user.password)
    
    # Create user
    db_user = User(
        email=user.email,
        name=user.name,  # This represents the userid
        role=user.role,
        password=hashed_password,
        location=None  # Not required anymore
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create JWT token
    token_data = {
        "sub": str(db_user.id),  # Subject (user ID)
        "email": db_user.email,
        "name": db_user.name,
        "role": db_user.role
    }
    access_token = create_access_token(data=token_data)
    
    return UserResponseWithToken(
        id=db_user.id,
        email=db_user.email,
        name=db_user.name,
        role=db_user.role,
        created_at=db_user.created_at,
        access_token=access_token,
        token_type="bearer"
    )

@router.post("/login", response_model=UserResponseWithToken)
def login_user(user_login: UserLogin, db: Session = Depends(get_db)):
    """Login user with email and password. Returns user data with JWT token."""
    # Find user by email
    db_user = db.query(User).filter(User.email == user_login.email).first()
    
    # Check if user exists
    if not db_user:
        raise HTTPException(
            status_code=402, 
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(user_login.password, db_user.password):
        raise HTTPException(
            status_code=402, 
            detail="Invalid email or password"
        )
    
    # Create JWT token
    token_data = {
        "sub": str(db_user.id),  # Subject (user ID)
        "email": db_user.email,
        "name": db_user.name,
        "role": db_user.role
    }
    access_token = create_access_token(data=token_data)
    
    return UserResponseWithToken(
        id=db_user.id,
        email=db_user.email,
        name=db_user.name,
        role=db_user.role,
        created_at=db_user.created_at,
        access_token=access_token,
        token_type="bearer"
    )

@router.get("/", response_model=List[UserResponse])
def get_users(
    role: Optional[str] = None,
    location: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """Get all users with optional filters."""
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
    
    if location:
        query = query.filter(User.location == location)
    
    users = query.all()
    
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            created_at=user.created_at
        )
        for user in users
    ]

@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """Get a specific user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at
    )

@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """Update a user."""
    db_user = db.query(User).filter(User.id == user_id).first()
    
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    
    # Hash password if it's being updated
    if "password" in update_data:
        update_data["password"] = hash_password(update_data["password"])
    
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        name=db_user.name,
        role=db_user.role,
        created_at=db_user.created_at
    )
