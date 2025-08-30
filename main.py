from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database.database import get_db, engine
from models import models
from api import documents, users, training, checklists, analytics, tts

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocBot API", description="API for F&B Training DocBot")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(checklists.router, prefix="/api/checklists", tags=["Checklists"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(tts.router, prefix="/api/tts", tags=["Text-to-Speech"])

@app.get("/")
def read_root():
    return {"message": "Welcome to DocBot API for F&B Training"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
