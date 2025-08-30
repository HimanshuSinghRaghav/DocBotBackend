# Backend Setup Guide

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/docbot

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Gemini API Configuration (Google AI)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Override default model
DEFAULT_AI_MODEL=gpt-3.5-turbo
```

## API Keys Required

1. **OpenAI API Key**: Required for embeddings and LLM responses
   - Get from: https://platform.openai.com/api-keys
   - Used for: Text embeddings and GPT responses

2. **Gemini API Key**: Alternative AI model
   - Get from: https://makersuite.google.com/app/apikey
   - Used for: Alternative LLM responses

3. **OpenRouter API Key**: Access to multiple AI models
   - Get from: https://openrouter.ai/keys
   - Used for: Access to various AI models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize the database:
```bash
python init_db.py
```

3. Start the server:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Features

- **AI-Powered RAG**: Uses multiple AI models with intelligent fallback
  - **Primary**: OpenAI GPT-3.5-turbo with embeddings
  - **Fallback 1**: Google Gemini Pro
  - **Fallback 2**: OpenRouter (access to multiple models)
  - **Final Fallback**: Simplified rule-based responses
- **Document Processing**: Supports PDF, DOCX, TXT, and Markdown files
- **Vector Search**: Semantic search using embeddings
- **Procedure Mode**: Step-by-step procedure generation
- **Source Citations**: Always cites the source documents
- **AI Model Status**: Check which AI model is currently being used

## Testing

1. Upload a document via the API
2. Check AI model status: `GET /api/documents/ai-status`
3. Query the document using the `/api/documents/query` endpoint
4. The system will now provide AI-generated responses based on the document content
5. The response will include which AI model was used

## API Endpoints

- `GET /api/documents/ai-status` - Check which AI model is currently active
- `POST /api/documents/query` - Query documents with AI-powered responses
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents
- `DELETE /api/documents/{id}` - Delete a document
