#!/usr/bin/env python3
"""
Database migration script to add category columns to content tables

This script adds 'category' columns to the following tables:
- quizzes
- questions  
- learning_modules
- learning_sessions
- session_content

Run this script after updating the models.py file to add the category columns.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SQLALCHEMY_DATABASE_URL

# SQL commands to add category columns
MIGRATION_SQL = [
    # Add category column to quizzes table
    """
    ALTER TABLE quizzes 
    ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'other';
    """,
    
    # Add category column to questions table  
    """
    ALTER TABLE questions 
    ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'other';
    """,
    
    # Add category column to learning_modules table
    """
    ALTER TABLE learning_modules 
    ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'other';
    """,
    
    # Add category column to learning_sessions table
    """
    ALTER TABLE learning_sessions 
    ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'other';
    """,
    
    # Add category column to session_content table
    """
    ALTER TABLE session_content 
    ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'other';
    """,
]

# SQL commands to create indexes for better performance
INDEX_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_quizzes_category ON quizzes(category);
    """,
    
    """
    CREATE INDEX IF NOT EXISTS idx_questions_category ON questions(category);
    """,
    
    """
    CREATE INDEX IF NOT EXISTS idx_learning_modules_category ON learning_modules(category);
    """,
    
    """
    CREATE INDEX IF NOT EXISTS idx_learning_sessions_category ON learning_sessions(category);
    """,
    
    """
    CREATE INDEX IF NOT EXISTS idx_session_content_category ON session_content(category);
    """,
]

# Update existing records to have meaningful categories (optional)
UPDATE_SQL = [
    """
    UPDATE quizzes 
    SET category = 'safety_guidance' 
    WHERE title ILIKE '%safety%' OR title ILIKE '%hazard%' OR title ILIKE '%protection%';
    """,
    
    """
    UPDATE quizzes 
    SET category = 'emergency_procedure' 
    WHERE title ILIKE '%emergency%' OR title ILIKE '%accident%' OR title ILIKE '%evacuation%';
    """,
    
    """
    UPDATE questions 
    SET category = 'safety_guidance' 
    WHERE question_text ILIKE '%safety%' OR question_text ILIKE '%hazard%' OR question_text ILIKE '%protection%';
    """,
    
    """
    UPDATE questions 
    SET category = 'emergency_procedure' 
    WHERE question_text ILIKE '%emergency%' OR question_text ILIKE '%accident%' OR question_text ILIKE '%evacuation%';
    """,
    
    """
    UPDATE learning_modules 
    SET category = 'safety_guidance' 
    WHERE title ILIKE '%safety%' OR description ILIKE '%safety%';
    """,
    
    """
    UPDATE learning_modules 
    SET category = 'emergency_procedure' 
    WHERE title ILIKE '%emergency%' OR description ILIKE '%emergency%';
    """,
]

def run_migration():
    """Run the database migration to add category columns."""
    print("🔄 Starting database migration to add content categories...")
    
    try:
        # Create database engine
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        with engine.connect() as connection:
            # Begin transaction
            trans = connection.begin()
            
            try:
                # Add category columns
                print("📊 Adding category columns to tables...")
                for i, sql in enumerate(MIGRATION_SQL, 1):
                    print(f"   {i}. Adding category column to table...")
                    connection.execute(text(sql))
                    
                print("✅ Category columns added successfully!")
                
                # Create indexes
                print("🔍 Creating indexes for better performance...")
                for i, sql in enumerate(INDEX_SQL, 1):
                    print(f"   {i}. Creating index...")
                    connection.execute(text(sql))
                    
                print("✅ Indexes created successfully!")
                
                # Update existing records with categories (optional)
                print("🏷️  Updating existing records with categories...")
                for i, sql in enumerate(UPDATE_SQL, 1):
                    result = connection.execute(text(sql))
                    print(f"   {i}. Updated {result.rowcount} records")
                    
                print("✅ Existing records updated successfully!")
                
                # Commit transaction
                trans.commit()
                print("🎉 Migration completed successfully!")
                
                # Print summary
                print("\n📋 Migration Summary:")
                print("   • Added 'category' columns to:")
                print("     - quizzes")
                print("     - questions")
                print("     - learning_modules")
                print("     - learning_sessions")
                print("     - session_content")
                print("   • Created indexes for performance")
                print("   • Updated existing records with categories")
                print("\n🔧 Category values:")
                print("   - safety_guidance: Safety rules, PPE, hazard identification")
                print("   - physical_work_process: Equipment operation, work procedures")
                print("   - emergency_procedure: Emergency response, first aid")
                print("   - other: General content (will not generate new content)")
                
                return True
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                raise e
                
    except SQLAlchemyError as e:
        print(f"❌ Database error during migration: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during migration: {e}")
        return False

def rollback_migration():
    """Rollback the migration by removing category columns."""
    print("🔄 Rolling back database migration...")
    
    rollback_sql = [
        "ALTER TABLE quizzes DROP COLUMN IF EXISTS category;",
        "ALTER TABLE questions DROP COLUMN IF EXISTS category;", 
        "ALTER TABLE learning_modules DROP COLUMN IF EXISTS category;",
        "ALTER TABLE learning_sessions DROP COLUMN IF EXISTS category;",
        "ALTER TABLE session_content DROP COLUMN IF EXISTS category;",
    ]
    
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        with engine.connect() as connection:
            trans = connection.begin()
            
            try:
                for sql in rollback_sql:
                    connection.execute(text(sql))
                    
                trans.commit()
                print("✅ Migration rolled back successfully!")
                return True
                
            except Exception as e:
                trans.rollback()
                raise e
                
    except Exception as e:
        print(f"❌ Error during rollback: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add content category columns to database")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_migration()
    else:
        success = run_migration()
    
    sys.exit(0 if success else 1)