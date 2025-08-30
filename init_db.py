from database.database import engine
from models import models

def init_database():
    """Initialize the database by creating all tables."""
    print("Creating database tables...")
    models.Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_database()
