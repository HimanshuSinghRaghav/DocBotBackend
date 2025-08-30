# Learning Session System 📚

## Overview

The Learning Session System automatically organizes large volumes of quiz questions (like your 400 questions from PDF) into structured, progressive learning modules and sessions.

## Key Features

### 🎯 **Smart Content Organization**
- **Automatic Grouping**: AI analyzes content and groups related lessons/questions into topics
- **Balanced Distribution**: Ensures each module has manageable amount of content
- **Topic-Based Modules**: Groups content by themes (e.g., "Hand Washing", "Temperature Control")

### 📖 **Progressive Learning Structure**
```
Document (PDF with 400 questions)
├── Module 1: Hand Washing (3 sessions)
│   ├── Session 1: Learning Content (lessons)
│   ├── Session 2: Practice Quiz (8 questions)
│   └── Session 3: Assessment (10 questions)
├── Module 2: Temperature Control (4 sessions)
│   ├── Session 1: Learning Content (lessons)
│   ├── Session 2: Basic Quiz (8 questions)
│   ├── Session 3: Advanced Quiz (10 questions)
│   └── Session 4: Final Assessment (12 questions)
└── Module N: Additional Topics...
```

### 🔐 **Progressive Unlocking**
- **Sequential Access**: Users must complete Session 1 before Session 2 unlocks
- **Module Prerequisites**: Can set module dependencies
- **Flexible Unlocking**: Custom unlock conditions supported

### 📊 **Comprehensive Progress Tracking**
- **Session Level**: Track attempts, scores, time spent
- **Module Level**: Overall progress percentage, completion status
- **User Level**: Learning path progress across all modules

## Database Models

### LearningModule
- Groups related content into topics
- Contains multiple learning sessions
- Tracks difficulty level and estimated duration

### LearningSession
- Individual learning units (5-10 questions or 2-3 lessons)
- Different types: `lesson`, `quiz`, `assessment`, `mixed`
- Configurable passing scores and attempt limits

### SessionContent
- Links lessons and quiz questions to sessions
- Maintains content order within sessions
- Supports different content types

### Progress Tracking
- **ModuleProgress**: User's progress through modules
- **SessionProgress**: Detailed session completion data

## API Endpoints

### Learning Path Management
```
GET /api/learning/documents/{document_id}/learning-path?user_id={user_id}
# Returns complete learning path with progress

GET /api/learning/modules/{module_id}?user_id={user_id}  
# Get module with all sessions and unlock status

GET /api/learning/sessions/{session_id}?user_id={user_id}
# Get session content (lessons or quiz questions)
```

### Session Interaction
```
POST /api/learning/sessions/{session_id}/attempt
# Submit quiz answers or complete lesson
Body: {
  "session_id": 123,
  "answers": {"1": "A", "2": "B", "3": "C"},
  "time_spent": 300
}

GET /api/learning/modules/{module_id}/progress?user_id={user_id}
# Get detailed progress information
```

## Content Generation Flow

### 1. PDF Processing
```python
# PDF uploaded → OCR extraction → Text chunks
pdf_content = extract_with_ocr(pdf_file)
chunks = split_into_chunks(pdf_content, size=1500, overlap=300)
```

### 2. AI Analysis
```python
# Each chunk analyzed individually
for chunk in chunks:
    analysis = ai_analyze_chunk(chunk)
    if analysis.has_learning_content:
        lessons.extend(analysis.learning_lessons)
    if analysis.has_quiz_content:
        questions.extend(analysis.quiz_questions)
```

### 3. Content Organization
```python
# Group content by topics
grouped_content = group_by_topics(lessons, questions)

# Create modules and sessions
for topic, content in grouped_content.items():
    module = create_module(topic, content)
    create_sessions_for_module(module, content)
```

## Frontend Implementation Guide

### Learning Dashboard
```tsx
function LearningDashboard({ documentId, userId }) {
  const { data: learningPath } = useLearningPath(documentId, userId);
  
  return (
    <div className="learning-modules">
      {learningPath.modules.map(module => (
        <ModuleCard 
          key={module.id}
          module={module}
          progress={module.user_progress}
          isUnlocked={module.is_unlocked}
        />
      ))}
    </div>
  );
}
```

### Session Player
```tsx
function SessionPlayer({ sessionId, userId }) {
  const { data: session } = useSession(sessionId, userId);
  const [answers, setAnswers] = useState({});
  
  if (session.session_type === 'lesson') {
    return <LessonPlayer content={session.session_content} />;
  } else {
    return <QuizPlayer questions={session.session_content} />;
  }
}
```

## Configuration Options

### Session Types
- **`lesson`**: Display educational content only
- **`quiz`**: Interactive questions with scoring
- **`assessment`**: High-stakes quiz with limited attempts
- **`mixed`**: Combination of lessons and questions

### Difficulty Levels
- **Beginner**: Simple concepts, shorter sessions
- **Intermediate**: Moderate complexity, longer sessions
- **Advanced**: Complex scenarios, comprehensive assessments

### Unlock Conditions
```json
{
  "type": "sequential",
  "requires_previous": true,
  "minimum_score": 70,
  "custom_conditions": []
}
```

## Benefits for Large Content Sets

### Before (400 questions in one quiz)
- ❌ Overwhelming for users
- ❌ No progress tracking
- ❌ All-or-nothing completion
- ❌ No learning structure

### After (Organized sessions)
- ✅ Manageable 5-10 question sessions
- ✅ Progressive difficulty increase
- ✅ Topic-based organization
- ✅ Detailed progress tracking
- ✅ Retry failed sections
- ✅ Immediate feedback
- ✅ Gamification potential

## Example Usage

```python
# After uploading a PDF with 400 questions
document = upload_pdf("safety_manual.pdf")

# System automatically creates:
modules = [
    {
        "title": "Hand Washing Procedures",
        "sessions": [
            {"type": "lesson", "content": "3 lessons"},
            {"type": "quiz", "questions": 8}
        ]
    },
    {
        "title": "Temperature Control", 
        "sessions": [
            {"type": "lesson", "content": "4 lessons"},
            {"type": "quiz", "questions": 10},
            {"type": "assessment", "questions": 12}
        ]
    }
    # ... more modules
]

# Users can then:
# 1. Start with Module 1, Session 1
# 2. Complete lessons at their own pace
# 3. Take manageable quizzes
# 4. Unlock next content progressively
# 5. Track their overall progress
```

## Next Steps

1. **Test the System**: Upload a large PDF and see the automatic organization
2. **Customize Settings**: Adjust session sizes, passing scores, attempts
3. **Frontend Integration**: Build the learning interface components
4. **Add Gamification**: Badges, streaks, leaderboards
5. **Analytics**: Track learning patterns and optimize content

The system transforms overwhelming content into an engaging, progressive learning experience! 🚀