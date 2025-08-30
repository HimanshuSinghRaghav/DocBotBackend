from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Literal
from datetime import datetime

class UserBase(BaseModel):
    email: str
    name: str  # This represents the userid
    role: Literal["admin", "shift_lead", "crew"]

class UserCreate(UserBase):
    password: str
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ["admin", "shift_lead", "crew"]
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v

class UserUpdate(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None
    role: Optional[Literal["admin", "shift_lead", "crew"]] = None
    password: Optional[str] = None

    @validator('role')
    def validate_role(cls, v):
        if v is not None:
            allowed_roles = ["admin", "shift_lead", "crew"]
            if v not in allowed_roles:
                raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponseWithToken(UserResponse):
    access_token: str
    token_type: str = "bearer"
