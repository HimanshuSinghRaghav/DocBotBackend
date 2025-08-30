#!/usr/bin/env python3
"""
Example Usage of the Learning Session System

This demonstrates how 400 quizzes from a PDF are automatically organized 
into structured learning modules and sessions.
"""

def example_learning_structure():
    """
    Example of how your 400 quizzes would be organized:
    
    Document: "Food Safety Training Manual" (400 quiz questions)
    
    After processing, the system creates:
    """
    
    example_structure = {
        "document_title": "Food Safety Training Manual",
        "total_questions": 400,
        "organized_structure": {
            "Module 1: Hand Washing and Personal Hygiene": {
                "estimated_duration": "45 minutes",
                "difficulty": "Beginner",
                "sessions": [
                    {
                        "session_1": "Hand Washing Procedures - Learning Content",
                        "type": "lesson",
                        "content": ["3 lessons covering proper techniques", "Key safety points", "Best practices"]
                    },
                    {
                        "session_2": "Hand Washing Quiz",
                        "type": "quiz", 
                        "content": ["8 quiz questions", "Passing score: 70%", "Max attempts: 3"]
                    }
                ]
            },
            "Module 2: Food Temperature Control": {
                "estimated_duration": "60 minutes", 
                "difficulty": "Intermediate",
                "sessions": [
                    {
                        "session_1": "Temperature Control - Learning Content",
                        "type": "lesson",
                        "content": ["4 lessons on temperature zones", "Critical control points", "Monitoring procedures"]
                    },
                    {
                        "session_2": "Temperature Control Quiz 1", 
                        "type": "quiz",
                        "content": ["10 quiz questions", "Focus on danger zones", "Passing score: 70%"]
                    },
                    {
                        "session_3": "Temperature Control Quiz 2",
                        "type": "quiz", 
                        "content": ["10 quiz questions", "Focus on monitoring", "Passing score: 70%"]
                    }
                ]
            },
            "Module 3: Cross-Contamination Prevention": {
                "estimated_duration": "50 minutes",
                "difficulty": "Intermediate", 
                "sessions": [
                    {
                        "session_1": "Cross-Contamination - Learning Content",
                        "type": "lesson",
                        "content": ["3 lessons on prevention methods", "Equipment cleaning", "Food separation"]
                    },
                    {
                        "session_2": "Cross-Contamination Assessment",
                        "type": "quiz",
                        "content": ["12 quiz questions", "Real-world scenarios", "Passing score: 70%"]
                    }
                ]
            },
            # ... more modules automatically created
            "Module N": "Additional modules created based on content analysis"
        }
    }
    
    return example_structure

def api_endpoints_examples():
    """
    API endpoints you can use to access the organized content:
    """
    
    endpoints = {
        "Get Learning Path": {
            "endpoint": "GET /api/learning/documents/{document_id}/learning-path?user_id={user_id}",
            "description": "Get complete learning path with modules and sessions",
            "response": "Shows progress, unlocked sessions, estimated time"
        },
        
        "Get Module with Sessions": {
            "endpoint": "GET /api/learning/modules/{module_id}?user_id={user_id}", 
            "description": "Get specific module with all its sessions",
            "response": "Module details, sessions, unlock status, progress"
        },
        
        "Start Session": {
            "endpoint": "GET /api/learning/sessions/{session_id}?user_id={user_id}",
            "description": "Get session content (lessons or quiz questions)",
            "response": "Session content, questions (without answers), instructions"
        },
        
        "Submit Session": {
            "endpoint": "POST /api/learning/sessions/{session_id}/attempt",
            "description": "Submit answers for a quiz session",
            "body": {"session_id": 123, "answers": {"1": "A", "2": "B"}, "time_spent": 300},
            "response": "Score, pass/fail, feedback, next session unlocked"
        },
        
        "Track Progress": {
            "endpoint": "GET /api/learning/modules/{module_id}/progress?user_id={user_id}",
            "description": "Get user's progress in a module", 
            "response": "Completion percentage, current session, status"
        }
    }
    
    return endpoints

def frontend_implementation_example():
    """
    How to implement this in your frontend:
    """
    
    frontend_flow = {
        "Learning Dashboard": {
            "description": "Show available modules with progress",
            "components": [
                "Module cards with progress bars",
                "Estimated time remaining", 
                "Difficulty indicators",
                "Lock/unlock status"
            ]
        },
        
        "Module View": {
            "description": "Show sessions within a module",
            "components": [
                "Session list with types (lesson/quiz)",
                "Progressive unlock indicators",
                "Progress tracking",
                "Continue/Start buttons"
            ]
        },
        
        "Session Player": {
            "description": "Present lesson content or quiz questions",
            "lesson_mode": [
                "Display lesson title and summary",
                "Show key points as cards",
                "Definitions popup/sidebar", 
                "Best practices highlights",
                "Next lesson button"
            ],
            "quiz_mode": [
                "Question counter (1 of 10)",
                "Timer (optional)",
                "Question with options",
                "Previous/Next navigation",
                "Submit button"
            ]
        },
        
        "Results Screen": {
            "description": "Show quiz results and next steps",
            "components": [
                "Score display",
                "Pass/fail status", 
                "Detailed feedback",
                "Retry button (if failed)",
                "Next session button (if passed)",
                "Module completion certificate (if module done)"
            ]
        }
    }
    
    return frontend_flow

if __name__ == "__main__":
    print("=== Learning Session System Example ===\n")
    
    structure = example_learning_structure()
    print("📚 Document Structure:")
    print(f"Original: {structure['total_questions']} quiz questions")
    print(f"Organized into: {len(structure['organized_structure'])} modules\n")
    
    for module_name, module_info in structure['organized_structure'].items():
        if isinstance(module_info, dict) and 'sessions' in module_info:
            print(f"📖 {module_name}")
            print(f"   Duration: {module_info['estimated_duration']}")
            print(f"   Difficulty: {module_info['difficulty']}")
            print(f"   Sessions: {len(module_info['sessions'])}")
            for session in module_info['sessions']:
                for session_name, session_type in session.items():
                    if session_name.startswith('session_'):
                        print(f"   - {session_type}")
            print()
    
    print("\n=== Key Benefits ===")
    benefits = [
        "✅ Progressive Learning: Users unlock sessions sequentially",
        "✅ Manageable Size: 5-10 questions per quiz session", 
        "✅ Topic-Based: Related content grouped together",
        "✅ Progress Tracking: Track completion at module and session level",
        "✅ Adaptive: Failed sessions can be retaken with feedback",
        "✅ Flexible: Different session types (lessons, quizzes, mixed)",
        "✅ Gamification: Unlocks, badges, and progress bars"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print(f"\n=== API Endpoints Available ===")
    endpoints = api_endpoints_examples()
    for name, info in endpoints.items():
        print(f"🔗 {name}: {info['endpoint']}")
        print(f"   Description: {info['description']}\n")