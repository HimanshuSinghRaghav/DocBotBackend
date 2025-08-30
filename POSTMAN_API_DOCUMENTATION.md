# DocBot API Documentation

Base URL: `http://localhost:8000`

## Authentication

The API uses OAuth2 password bearer authentication. Include the bearer token in the Authorization header for protected endpoints.

## Users API

### Create User
- **Endpoint**: `POST /api/users/`
- **Description**: Create a new user
- **Request Body**:
  ```json
  {
    "email": "string",
    "name": "string",
    "role": "string",  // Crew, Shift Lead, Manager
    "location": "string"
  }
  ```
- **Response**: User object with ID and creation timestamp

### Get Users
- **Endpoint**: `GET /api/users/`
- **Description**: Get all users with optional filters
- **Query Parameters**:
  - `role` (optional): Filter by user role
  - `location` (optional): Filter by user location
- **Response**: Array of user objects

### Get User by ID
- **Endpoint**: `GET /api/users/{user_id}`
- **Description**: Get a specific user by ID
- **Response**: User object

### Update User
- **Endpoint**: `PUT /api/users/{user_id}`
- **Description**: Update user information
- **Request Body**:
  ```json
  {
    "email": "string",  // optional
    "name": "string",   // optional
    "role": "string",   // optional
    "location": "string" // optional
  }
  ```
- **Response**: Updated user object

## Documents API

### Upload Document
- **Endpoint**: `POST /api/documents/upload`
- **Description**: Upload and process a new document
- **Form Data**:
  - `file`: Document file
  - `title`: Document title
  - `document_type`: Type of document
  - `version`: Document version
  - `effective_date`: Effective date (YYYY-MM-DD)
  - `language` (optional): Document language (default: "en")
- **Response**: Document object with processing status

### Get Documents
- **Endpoint**: `GET /api/documents/`
- **Description**: Get all documents with optional filters
- **Query Parameters**:
  - `document_type` (optional): Filter by document type
  - `language` (optional): Filter by language
- **Response**: Array of document objects

### Get Document by ID
- **Endpoint**: `GET /api/documents/{document_id}`
- **Description**: Get a specific document by ID
- **Response**: Document object

### Delete Document
- **Endpoint**: `DELETE /api/documents/{document_id}`
- **Description**: Delete a document and its associated file
- **Response**: Success message

### Query Documents
- **Endpoint**: `POST /api/documents/query`
- **Description**: Query documents using RAG (Retrieval-Augmented Generation)
- **Request Body**:
  ```json
  {
    "query": "string",
    "document_ids": [1, 2],  // optional
    "procedure_mode": false,  // optional
    "language": "en"         // optional
  }
  ```
- **Response**: Query results with AI-generated answer and sources

### Get AI Status
- **Endpoint**: `GET /api/documents/ai-status`
- **Description**: Get current AI model status
- **Response**: AI configuration status

## Text-to-Speech API

### Convert Text to Speech
- **Endpoint**: `POST /api/tts/convert`
- **Description**: Convert text to speech using Murf API
- **Request Body**:
  ```json
  {
    "text": "string",
    "voice_id": "en-US-julia"  // optional, defaults to en-US-julia
  }
  ```
- **Response**: 
  ```json
  {
    "audio_base64": "string",  // Base64 encoded audio data
    "message": "string",
    "success": true
  }
  ```

### Get Available Voices
- **Endpoint**: `GET /api/tts/voices`
- **Description**: Get list of available Murf voices
- **Response**: Array of voice objects

## Training API

### Create Quiz
- **Endpoint**: `POST /api/training/quizzes`
- **Description**: Create a new quiz
- **Request Body**:
  ```json
  {
    "title": "string",
    "document_id": 1
  }
  ```
- **Response**: Quiz object

### Get Quizzes
- **Endpoint**: `GET /api/training/quizzes`
- **Description**: Get all quizzes with optional document filter
- **Query Parameters**:
  - `document_id` (optional): Filter by document ID
- **Response**: Array of quiz objects

### Create Question
- **Endpoint**: `POST /api/training/quizzes/{quiz_id}/questions`
- **Description**: Create a new question for a quiz
- **Request Body**:
  ```json
  {
    "question_text": "string",
    "question_type": "string",  // MCQ, True/False
    "options": {
      "A": "Option text",
      "B": "Option text"
    },
    "correct_answer": "string",
    "explanation": "string",
    "source_chunk_id": 1  // optional
  }
  ```
- **Response**: Question object

### Get Questions
- **Endpoint**: `GET /api/training/quizzes/{quiz_id}/questions`
- **Description**: Get all questions for a specific quiz
- **Response**: Array of question objects

### Record Quiz Attempt
- **Endpoint**: `POST /api/training/attempts`
- **Description**: Record a quiz attempt
- **Request Body**:
  ```json
  {
    "user_id": 1,
    "quiz_id": 1,
    "score": 85.5,
    "answers": {
      "question_id": "selected_answer"
    }
  }
  ```
- **Response**: Quiz attempt object

### Create/Update Training Record
- **Endpoint**: `POST /api/training/records`
- **Description**: Create or update a training record
- **Request Body**:
  ```json
  {
    "user_id": 1,
    "document_id": 1,
    "status": "string",  // Not Started, In Progress, Completed
    "progress": 75.5     // 0-100%
  }
  ```
- **Response**: Training record object

### Get Training Records
- **Endpoint**: `GET /api/training/records`
- **Description**: Get training records with optional filters
- **Query Parameters**:
  - `user_id` (optional): Filter by user ID
  - `document_id` (optional): Filter by document ID
  - `status` (optional): Filter by status
- **Response**: Array of training record objects

## Checklists API

### Create Checklist
- **Endpoint**: `POST /api/checklists/`
- **Description**: Create a new checklist
- **Request Body**:
  ```json
  {
    "title": "string",
    "description": "string",
    "frequency": "string",  // Daily, Weekly, Monthly
    "items": [
      {
        "id": "1",
        "text": "Check item description",
        "type": "checkbox"
      }
    ],
    "document_id": 1  // optional
  }
  ```
- **Response**: Checklist object

### Get Checklists
- **Endpoint**: `GET /api/checklists/`
- **Description**: Get all checklists with optional filters
- **Query Parameters**:
  - `frequency` (optional): Filter by frequency
  - `document_id` (optional): Filter by document ID
- **Response**: Array of checklist objects

### Get Checklist by ID
- **Endpoint**: `GET /api/checklists/{checklist_id}`
- **Description**: Get a specific checklist by ID
- **Response**: Checklist object

### Record Checklist Completion
- **Endpoint**: `POST /api/checklists/completions`
- **Description**: Record a checklist completion
- **Request Body**:
  ```json
  {
    "checklist_id": 1,
    "user_id": 1,
    "responses": {
      "item_id": {
        "checked": true,
        "notes": "string"
      }
    },
    "attestation": true
  }
  ```
- **Response**: Checklist completion object

### Get Checklist Completions
- **Endpoint**: `GET /api/checklists/completions`
- **Description**: Get checklist completions with optional filters
- **Query Parameters**:
  - `checklist_id` (optional): Filter by checklist ID
  - `user_id` (optional): Filter by user ID
- **Response**: Array of checklist completion objects

## Analytics API

### Get Training Completion Stats
- **Endpoint**: `GET /api/analytics/training/completion`
- **Description**: Get training completion statistics
- **Query Parameters**:
  - `document_id` (optional): Filter by document ID
  - `role` (optional): Filter by user role
  - `location` (optional): Filter by location
- **Response**: Training completion statistics

### Get Training Scores
- **Endpoint**: `GET /api/analytics/training/scores`
- **Description**: Get quiz score statistics
- **Query Parameters**:
  - `quiz_id` (optional): Filter by quiz ID
  - `role` (optional): Filter by user role
  - `location` (optional): Filter by location
- **Response**: Quiz score statistics

### Get Checklist Adherence
- **Endpoint**: `GET /api/analytics/checklists/adherence`
- **Description**: Get checklist adherence statistics
- **Query Parameters**:
  - `checklist_id` (optional): Filter by checklist ID
  - `role` (optional): Filter by user role
  - `location` (optional): Filter by location
  - `days` (optional): Number of days to analyze (default: 30)
- **Response**: Checklist adherence statistics

### Get Dashboard Summary
- **Endpoint**: `GET /api/analytics/dashboard/summary`
- **Description**: Get a summary of key metrics for the dashboard
- **Response**: Comprehensive dashboard statistics including:
  - User statistics
  - Training statistics
  - Quiz statistics
  - Checklist statistics
  - Document statistics

## Error Responses

All endpoints may return the following error responses:

- **400 Bad Request**: Invalid request parameters or body
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Requested resource not found
- **500 Internal Server Error**: Server-side error

## Rate Limiting

The API implements rate limiting to prevent abuse. Clients should handle 429 Too Many Requests responses by implementing exponential backoff.