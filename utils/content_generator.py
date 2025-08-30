import os
import json
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
import math

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai

from models.models import Document, Quiz, Question, TrainingRecord, Chunk
from utils.rag_engine import RAGEngine

# Define schema for AI responses
CHUNK_ANALYSIS_SCHEMA = """
{
  "has_learning_content": boolean,  // Whether this chunk contains material suitable for learning content
  "has_quiz_content": boolean,     // Whether this chunk contains material suitable for quiz questions
  "learning_lessons": [             // Array of lessons if has_learning_content is true, empty array otherwise
    {
      "title": string,              // Title of the lesson
      "summary": string,            // Brief summary of the lesson (2-3 sentences)
      "key_points": [string],       // Array of key points (3-5 items)
      "definitions": {              // Object with terms and definitions (if any)
        "term1": "definition1",
        "term2": "definition2"
      },
      "best_practices": [string]    // Array of best practices or tips (if any)
    }
  ],
  "quiz_questions": [               // Array of questions if has_quiz_content is true, empty array otherwise
    {
      "question_text": string,       // The actual question
      "question_type": string,       // Either "MCQ" or "TRUE_FALSE"
      "options": {                   // Options for the question
        "A": string,
        "B": string,
        "C": string,                 // Optional for MCQ
        "D": string                  // Optional for MCQ
      },
      "correct_answer": string,      // The correct option (A, B, C, or D)
      "explanation": string,         // Explanation of why the answer is correct
      "source_text": string         // Source text from the chunk that supports this question
    }
  ]
}
"""

class ContentGenerator:
    def __init__(self, db: Session):
        self.db = db
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Initialize AI models with fallback options - same approach as RAGEngine
        self.llm = None
        self.gemini_model = None
        self.openrouter_model = None
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,  # Higher temperature for more creative generation
                    api_key=self.openai_api_key
                )
                print("Using OpenAI for content generation")
            except Exception as e:
                print(f"Error initializing OpenAI for content generation: {e}")
        
        # Try Gemini as fallback
        if not self.llm and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                print("Using Gemini for content generation")
            except Exception as e:
                print(f"Error initializing Gemini for content generation: {e}")
        
        # Try OpenRouter as fallback
        if not self.llm and not self.gemini_model and self.openrouter_api_key:
            try:
                self.openrouter_model = ChatOpenAI(
                    model="openai/gpt-3.5-turbo",
                    temperature=0.7,
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=self.openrouter_api_key
                )
                print("Using OpenRouter for content generation")
            except Exception as e:
                print(f"Error initializing OpenRouter for content generation: {e}")
                
    def generate_learning_content(self, document_id: int) -> Tuple[bool, str]:
        """
        Generate learning content from a document by processing individual chunks
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the document
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False, f"Document with ID {document_id} not found"
            
            # Prepare document content as chunks
            rag_engine = RAGEngine(self.db)
            document_chunks = self._get_document_content(document_id, rag_engine)
            
            if not document_chunks:
                return False, f"Could not retrieve content for document {document_id}"
            
            print(f"Processing document with {len(document_chunks)} chunks")
            
            # Validate document content quality
            valid_chunks = []
            for chunk in document_chunks:
                content = chunk.get("content", "")
                # Skip placeholder, binary data, or very small chunks
                is_valid = (
                    content and 
                    len(content.strip()) > 100 and 
                    not ("placeholder" in content and "PDF content from" in content) and
                    self._is_readable_text(content)  # Check if content is actually readable text
                )
                if is_valid:
                    valid_chunks.append(chunk)
                    
            if not valid_chunks and document_chunks:
                print(f"Warning: No valid content chunks in document {document_id} - document may not have been properly extracted")
                document = self.db.query(Document).filter(Document.id == document_id).first()
                if document and document.doc_metadata is None:
                    document.doc_metadata = {}
                if document:
                    document.doc_metadata["processing_warning"] = "No valid content chunks detected. PDF extraction may have failed."
                    self.db.commit()
                return False, f"No valid content found in document {document_id}"
            
            # Process each chunk individually with OpenRouter AI
            all_lessons = []
            all_questions = []
            
            print(f"Processing {len(valid_chunks)} valid chunks individually")
            for i, chunk in enumerate(valid_chunks):
                print(f"\nProcessing chunk {i+1}/{len(valid_chunks)}")
                chunk_content = chunk.get("content", "")
                chunk_metadata = chunk.get("metadata", {})
                
                # For each chunk, determine if it contains material for learning content and/or quiz
                chunk_lessons, chunk_questions = self._process_individual_chunk(
                    document.title, 
                    chunk_content, 
                    chunk_metadata,
                    chunk_id=chunk.get("chunk_id")
                )
                
                # Collect results
                if chunk_lessons:
                    all_lessons.extend(chunk_lessons)
                if chunk_questions:
                    all_questions.extend(chunk_questions)
                    
                # Give a progress update
                print(f"  - Found {len(chunk_lessons)} lessons and {len(chunk_questions)} questions in chunk {i+1}")
            
            # Create consolidated quiz data structure
            quiz_data = {
                "quiz_title": f"Quiz for {document.title}",
                "questions": all_questions
            }
            
            # Save lessons and quiz to database
            success, message = self._save_content_to_db(document, all_lessons, quiz_data)
            
            return success, message
            
        except Exception as e:
            traceback.print_exc()
            return False, f"Error generating learning content: {str(e)}"
    
    def _get_document_content(self, document_id: int, rag_engine: RAGEngine) -> List[Dict[str, Any]]:
        """Get document content from cached chunks or database as a list of chunks with metadata"""
        try:
            # Try to load cached content first
            cached_chunks, _ = rag_engine._load_cached_embeddings(document_id)
            if cached_chunks:
                # Convert to dict with content and metadata
                return [{
                    "content": chunk,
                    "metadata": {"index": i, "source": "cached_chunk"}
                } for i, chunk in enumerate(cached_chunks)]
            
            # Fallback to database chunks
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document or not document.chunks:
                return []
            
            # Get chunks with metadata
            return [{
                "content": chunk.content,
                "metadata": chunk.chunk_metadata or {"index": i},
                "chunk_id": chunk.id
            } for i, chunk in enumerate(document.chunks)]
            
        except Exception as e:
            print(f"Error getting document content: {e}")
            return []
    
    def _is_readable_text(self, content: str) -> bool:
        """
        Check if content is readable text (not binary data or placeholder)
        
        Args:
            content: Content to check
            
        Returns:
            True if content appears to be readable text
        """
        if not content or len(content.strip()) < 10:
            return False
            
        # Check for binary data indicators
        binary_indicators = [b'\x00', b'\xFF', '%PDF', '\x1f\x8b']  # Common binary patterns
        content_bytes = content.encode() if isinstance(content, str) else content
        
        for indicator in binary_indicators:
            if isinstance(indicator, str):
                if indicator in content:
                    return False
            else:
                if indicator in content_bytes:
                    return False
        
        # Check if content has reasonable text characteristics
        try:
            # Count printable characters
            printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
            total_chars = len(content)
            
            if total_chars == 0:
                return False
                
            printable_ratio = printable_chars / total_chars
            
            # Content should be mostly printable
            if printable_ratio < 0.7:
                return False
                
            # Check for reasonable word-like content
            words = re.findall(r'\b\w+\b', content)
            if len(words) < 5:  # Should have at least some words
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking if content is readable: {e}")
            return False
    
    def _process_individual_chunk(
        self, 
        document_title: str, 
        chunk_content: str, 
        chunk_metadata: Dict[str, Any], 
        chunk_id: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process an individual chunk to determine if it contains learning content or quiz material
        
        Args:
            document_title: Title of the document
            chunk_content: Content of the chunk to analyze
            chunk_metadata: Metadata about the chunk
            chunk_id: Optional chunk ID for reference
            
        Returns:
            Tuple of (lessons, questions) found in this chunk
        """
        try:
            # Check if chunk has enough content for analysis
            if len(chunk_content.strip()) < 50:
                print(f"Chunk too short for analysis: {len(chunk_content)} characters")
                return [], []
            
            # Create the analysis prompt
            system_prompt = f"""
You are an expert content analyzer for training materials. Your task is to analyze document chunks and determine if they contain material suitable for learning content or quiz questions.

Analyze the provided chunk and respond with JSON in this exact format:
{CHUNK_ANALYSIS_SCHEMA}

Important guidelines:
1. Only set has_learning_content to true if the chunk contains substantive educational material
2. Only set has_quiz_content to true if the chunk contains specific facts, procedures, or concepts that can be tested
3. If the chunk doesn't contain meaningful learning material, set both to false and return empty arrays
4. Generate 1-3 lessons maximum for learning content
5. Generate 2-5 quiz questions maximum for quiz content
6. Ensure all information is directly based on the chunk content
"""

            user_prompt = f"""
Document: {document_title}
Chunk Content:
{chunk_content}

Analyze this chunk and determine if it contains material that can be used as learning content or quiz questions. If the content is not substantial enough for learning or testing (like headers, footers, random text, etc.), respond with has_learning_content: false and has_quiz_content: false.

Provide your analysis in the specified JSON format.
"""

            # Get AI response
            response = self._generate_ai_response(system_prompt, user_prompt)
            
            if not response:
                print(f"No response from AI for chunk {chunk_id}")
                return [], []
            
            # Parse the JSON response
            try:
                # Extract JSON from response
                json_text = self._extract_json_from_text(response)
                if not json_text:
                    print(f"No JSON found in AI response for chunk {chunk_id}")
                    return [], []
                
                analysis = json.loads(json_text)
                
                # Extract lessons and questions
                lessons = analysis.get('learning_lessons', [])
                questions = analysis.get('quiz_questions', [])
                
                # Add chunk metadata to lessons
                for lesson in lessons:
                    lesson['source_chunk_id'] = chunk_id
                    lesson['source_metadata'] = chunk_metadata
                
                # Add chunk metadata to questions
                for question in questions:
                    question['source_chunk_id'] = chunk_id
                    question['source_metadata'] = chunk_metadata
                    # Ensure source_text is included
                    if 'source_text' not in question:
                        question['source_text'] = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
                
                print(f"Chunk analysis: {len(lessons)} lessons, {len(questions)} questions")
                return lessons, questions
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON from AI response for chunk {chunk_id}: {e}")
                print(f"Response was: {response[:500]}...")
                
                # Try to fix JSON formatting
                fixed_json = self._try_fix_json(response)
                if fixed_json:
                    try:
                        analysis = json.loads(fixed_json)
                        lessons = analysis.get('learning_lessons', [])
                        questions = analysis.get('quiz_questions', [])
                        
                        # Add metadata as above
                        for lesson in lessons:
                            lesson['source_chunk_id'] = chunk_id
                            lesson['source_metadata'] = chunk_metadata
                        
                        for question in questions:
                            question['source_chunk_id'] = chunk_id
                            question['source_metadata'] = chunk_metadata
                            if 'source_text' not in question:
                                question['source_text'] = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
                        
                        print(f"Fixed JSON parsing - Chunk analysis: {len(lessons)} lessons, {len(questions)} questions")
                        return lessons, questions
                    except Exception as fix_error:
                        print(f"Error even after fixing JSON for chunk {chunk_id}: {fix_error}")
                
                return [], []
                
        except Exception as e:
            print(f"Error processing individual chunk {chunk_id}: {e}")
            traceback.print_exc()
            return [], []
        """
        Split document chunks into batches for processing
        
        Args:
            chunks: List of document chunks
            max_batch_size: Maximum number of chunks per batch
            
        Returns:
            List of chunk batches
        """
        batches = []
        total_chunks = len(chunks)
        
        # Create batches with up to max_batch_size chunks
        for i in range(0, total_chunks, max_batch_size):
            batch = chunks[i:i + max_batch_size]
            batches.append(batch)
        
        # Make sure batches maintain semantic coherence when possible
        # (currently just using simple grouping, could be improved)
        
        print(f"Split document into {len(batches)} batches for processing")
        return batches
    
    def _generate_lessons_map_reduce(self, title: str, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Generate learning lessons using map-reduce approach
        
        Args:
            title: Document title
            chunks: List of document content chunks
            
        Returns:
            List of lessons with sections
        """
        if not chunks:
            return [{
                "title": f"Learning content for {title}",
                "summary": "No content available to generate lessons.",
                "key_points": ["Document appears to be empty"],
                "definitions": {},
                "best_practices": ["Ensure document contains content"]
            }]
        
        # If document is small enough, process it directly
        if len(chunks) <= 3:  # Small document optimization
            joined_content = "\n\n".join([chunk["content"] for chunk in chunks])
            return self._generate_lessons_from_content(title, joined_content, chunks)
        
        # For larger documents, use map-reduce approach
        # MAP: Process document chunks in batches
        batches = self._split_chunks_into_batches(chunks)
        batch_results = []
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1} of {len(batches)}")
            batch_content = "\n\n".join([chunk["content"] for chunk in batch])
            batch_title = f"{title} (Part {i+1})"
            batch_lessons = self._generate_lessons_from_content(batch_title, batch_content, batch)
            batch_results.extend(batch_lessons)
        
        # REDUCE: Combine and consolidate results
        if len(batch_results) <= 5:  # If few enough lessons, return directly
            return batch_results
        
        # Too many lessons - need to consolidate/summarize
        return self._consolidate_lessons(title, batch_results)
    
    def _generate_lessons_from_content(self, title: str, content: str, source_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate learning lessons from a single content chunk
        
        Args:
            title: Document title or section
            content: Document content
            source_chunks: Original chunks with metadata for reference
            
        Returns:
            List of lessons with sections
        """
        # Check if content contains enough material for meaningful lessons
        if len(content.strip()) < 100:  # Very short content
            print(f"Content for {title} too short for meaningful lessons")
            return [{
                "title": f"Learning content for {title}",
                "summary": "Insufficient content for detailed lessons.",
                "key_points": ["Review the full document for complete information"],
                "definitions": {},
                "best_practices": ["Refer to the complete document for detailed guidance"],
                "source_chunks": [chunk.get("metadata", {}) for chunk in source_chunks]
            }]
        system_prompt = """You are an expert training content creator. Your task is to create structured learning content from the document provided.
Break down the content into clear, organized lessons that would be easy for employees to understand and learn from.

For each major section or topic, create a lesson with:
1. A descriptive title
2. A brief summary
3. Key learning points
4. Important definitions or terms
5. Best practices or tips

Format your response as a JSON array where each object has:
- "title": The lesson title
- "summary": A concise summary of the lesson (2-3 sentences)
- "key_points": Array of the most important points (3-5 items)
- "definitions": Object with term:definition pairs for important terminology
- "best_practices": Array of best practices or tips (2-4 items)

Focus on extracting the most valuable and practical information from the document."""

        # Limit individual content chunks to avoid token limits
        max_content_length = 6000
        if len(content) > max_content_length:
            content = content[:max_content_length]

        # Extract any section headings or key terms for better context
        key_terms = self._extract_key_terms(content)
        
        user_prompt = f"""Document Section: {title}

Document Content:
{content}

Key Terms: {', '.join(key_terms) if key_terms else 'None identified'}

Please generate 1-2 focused lessons that cover the key information in this section of the document. 
Ensure the lessons directly relate to the specific content provided.
Only include information that is explicitly mentioned in or can be directly inferred from the document."""

        try:
            result = self._generate_ai_response(system_prompt, user_prompt)
            
            # Parse the JSON response with improved error handling
            try:
                # Extract JSON from the response (in case there's explanatory text)
                json_text = self._extract_json_from_text(result)
                if not json_text:
                    print("No JSON found in response")
                    raise ValueError("No JSON found in response")
                    
                lessons = json.loads(json_text)
                if not isinstance(lessons, list):
                    lessons = [lessons]  # Convert to list if single object
                    
                # Add source chunk references to each lesson
                for lesson in lessons:
                    if "source_chunks" not in lesson:
                        lesson["source_chunks"] = [chunk.get("metadata", {}) for chunk in source_chunks]
                
                return lessons
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON from AI response for lessons: {e}")
                print(f"Response was: {result[:500]}...")
                
                # Try to fix common JSON formatting issues
                fixed_json = self._try_fix_json(result)
                if fixed_json:
                    try:
                        lessons = json.loads(fixed_json)
                        if not isinstance(lessons, list):
                            lessons = [lessons]
                        print("Successfully parsed JSON after fixing format")
                        return lessons
                    except:
                        pass
                
                # Fallback to empty structure
                return [{
                    "title": f"Learning content for {title}",
                    "summary": "This is automatically generated learning content.",
                    "key_points": ["Review the document carefully", "Ask your manager if you have questions"],
                    "definitions": {},
                    "best_practices": ["Follow all safety procedures", "Complete all training materials"]
                }]
                
        except Exception as e:
            print(f"Error generating lessons: {e}")
            # Return a basic fallback lesson
            return [{
                "title": f"Learning content for {title}",
                "summary": "This is automatically generated learning content.",
                "key_points": ["Review the document carefully", "Ask your manager if you have questions"],
                "definitions": {},
                "best_practices": ["Follow all safety procedures", "Complete all training materials"]
            }]
    
    def _consolidate_lessons(self, title: str, lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate many lessons into a smaller set
        
        Args:
            title: Document title
            lessons: List of all lessons generated
            
        Returns:
            Consolidated list of lessons
        """
        # Convert lessons to JSON for the prompt
        lessons_json = json.dumps(lessons, indent=2)
        
        system_prompt = """You are an expert training content curator. Your task is to consolidate and organize many lessons into a smaller, more focused set.

You'll receive a list of lessons that were generated from different parts of a document.
Consolidate these into 5-7 focused, comprehensive lessons by:
1. Merging similar topics
2. Eliminating duplicates
3. Prioritizing the most important information
4. Ensuring comprehensive coverage of the original material

Maintain the same JSON structure for each lesson:
- "title": A clear, descriptive title
- "summary": A concise summary (2-3 sentences)
- "key_points": Array of important points (3-5 items)
- "definitions": Object with term:definition pairs
- "best_practices": Array of best practices (2-4 items)

Return the consolidated lessons as a JSON array."""

        user_prompt = f"""Document Title: {title}

Original Lessons:
{lessons_json}

Please consolidate these lessons into 5-7 comprehensive, non-duplicative lessons that capture the most important information."""

        try:
            result = self._generate_ai_response(system_prompt, user_prompt)
            
            # Parse the JSON response with improved error handling
            try:
                # Extract JSON from the response
                json_text = self._extract_json_from_text(result)
                if not json_text:
                    print("No JSON found in consolidation response")
                    raise ValueError("No JSON found in response")
                    
                consolidated_lessons = json.loads(json_text)
                if not isinstance(consolidated_lessons, list):
                    consolidated_lessons = [consolidated_lessons]
                return consolidated_lessons
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON from consolidation response: {e}")
                # Try to fix common JSON formatting issues
                fixed_json = self._try_fix_json(result)
                if fixed_json:
                    try:
                        fixed_lessons = json.loads(fixed_json)
                        if not isinstance(fixed_lessons, list):
                            fixed_lessons = [fixed_lessons]
                        print("Successfully parsed JSON after fixing format")
                        return fixed_lessons
                    except:
                        pass
                        
                # If consolidation fails, return top 5 original lessons
                return lessons[:5]
                
        except Exception as e:
            print(f"Error consolidating lessons: {e}")
            # Return top 5 original lessons if consolidation fails
            return lessons[:5]
    
    def _generate_quiz_map_reduce(self, title: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Generate quiz questions using map-reduce approach
        
        Args:
            title: Document title
            chunks: List of document content chunks
            
        Returns:
            Quiz data structure with questions
        """
        if not chunks:
            return {
                "quiz_title": f"Quiz for {title}",
                "questions": [{
                    "question_text": f"No content available for quiz on {title}.",
                    "question_type": "MCQ",
                    "options": {"A": "True", "B": "False"},
                    "correct_answer": "B",
                    "explanation": "No document content was provided."
                }]
            }
        
        # If document is small enough, process it directly
        if len(chunks) <= 3:  # Small document optimization
            joined_content = "\n\n".join([chunk["content"] for chunk in chunks])
            return self._generate_quiz_from_content(title, joined_content, chunks)
        
        # For larger documents, use map-reduce approach
        # MAP: Process document chunks in batches to generate questions
        batches = self._split_chunks_into_batches(chunks)
        all_questions = []
        
        for i, batch in enumerate(batches):
            print(f"Processing quiz batch {i+1} of {len(batches)}")
            batch_content = "\n\n".join([chunk["content"] for chunk in batch])
            batch_title = f"{title} (Part {i+1})"
            batch_quiz = self._generate_quiz_from_content(batch_title, batch_content, batch)
            all_questions.extend(batch_quiz.get("questions", []))
        
        # REDUCE: Select the best questions
        # If we have too many questions, select a diverse subset
        if len(all_questions) > 15:
            final_questions = self._select_diverse_questions(all_questions, 15)
        else:
            final_questions = all_questions
        
        return {
            "quiz_title": f"Comprehensive Quiz on {title}",
            "questions": final_questions
        }
    
    def _extract_key_terms(self, content: str, max_terms: int = 5) -> List[str]:
        """
        Extract key terms from content for better context
        
        Args:
            content: Document content
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of key terms
        """
        # Simple implementation - look for capitalized phrases and common patterns
        terms = []
        
        # Look for capitalized phrases (potential terms)
        cap_pattern = re.compile(r'\b[A-Z][A-Za-z0-9]+(\s+[A-Z][A-Za-z0-9]+)*\b')
        cap_matches = cap_pattern.findall(content)
        if cap_matches:
            # Filter out very short or very long terms
            terms.extend([term for term in cap_matches if 3 <= len(term) <= 30])[:max_terms]
            
        # Look for "defined" terms (X: Y or X - Y patterns)
        def_pattern = re.compile(r'\b([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+){0,3})(?:\:|\s+\-\s+)\s+')
        def_matches = def_pattern.findall(content)
        if def_matches:
            terms.extend([term.strip() for term in def_matches if 3 <= len(term) <= 30])[:max_terms]
            
        # Deduplicate and limit
        unique_terms = []
        for term in terms:
            if term not in unique_terms:
                unique_terms.append(term)
                if len(unique_terms) >= max_terms:
                    break
                    
        return unique_terms
    
    def _generate_quiz_from_content(self, title: str, content: str, source_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate quiz questions from a single content chunk
        
        Args:
            title: Document title or section
            content: Document content
            source_chunks: Original chunks with metadata for reference
            
        Returns:
            Quiz data structure with questions
        """
        # Check if content contains enough material for meaningful questions
        if len(content.strip()) < 100:  # Very short content
            print(f"Content for {title} too short for meaningful quiz questions")
            return {
                "quiz_title": f"Quiz for {title}",
                "questions": []
            }
        """
        Generate quiz questions from a single content chunk
        
        Args:
            title: Document title or section
            content: Document content
            
        Returns:
            Quiz data structure with questions
        """
        system_prompt = """You are an expert training assessment creator. Your task is to create a quiz based on the document provided.
Create questions that test understanding of key concepts, procedures, and important details from the document.

IMPORTANT: Only create questions that can be directly answered from the provided text. Do not make up information.
If the text doesn't contain enough specific information for good questions, create fewer questions or none.

Create a mix of multiple choice questions and true/false questions.

Format your response as a JSON object with:
- "quiz_title": A descriptive title for the quiz
- "questions": An array of question objects, where each question has:
  - "question_text": The actual question
  - "question_type": Either "MCQ" or "TRUE_FALSE"
  - "options": For MCQs, an object with options like {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"}, or for TRUE_FALSE: {"A": "True", "B": "False"}
  - "correct_answer": The letter of the correct option (e.g., "A", "B", "C", or "D")
  - "explanation": Brief explanation of why the answer is correct
  - "source_text": The specific text from the document that supports this question/answer

Create 3-5 questions that test understanding of the most important content from this specific section of the document.
For each question, include a direct quote or reference to where in the document the answer can be found."""

        # Limit individual content chunks to avoid token limits
        max_content_length = 6000
        if len(content) > max_content_length:
            content = content[:max_content_length]

        # Extract any section headings or key terms for better context
        key_terms = self._extract_key_terms(content)
        
        user_prompt = f"""Document Section: {title}

Document Content:
{content}

Key Terms: {', '.join(key_terms) if key_terms else 'None identified'}

Please generate 3-5 quiz questions that test understanding of the key concepts from this section of the document.
Ensure each question directly relates to specific information in the text.
If the content doesn't contain enough specific material for good questions, it's better to generate fewer questions or no questions.

For each question, include the specific source text from the document that supports the answer."""

        try:
            result = self._generate_ai_response(system_prompt, user_prompt)
            
            # Parse the JSON response with improved error handling
            try:
                # Extract JSON from the response
                json_text = self._extract_json_from_text(result)
                if not json_text:
                    print("No JSON found in quiz response")
                    raise ValueError("No JSON found in response")
                    
                quiz_data = json.loads(json_text)
                return quiz_data
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON from AI response for quiz: {e}")
                
                # Try to fix common JSON formatting issues
                fixed_json = self._try_fix_json(result)
                if fixed_json:
                    try:
                        quiz_data = json.loads(fixed_json)
                        print("Successfully parsed JSON after fixing format")
                        return quiz_data
                    except:
                        pass
                
                # Fallback to empty structure
                return {
                    "quiz_title": f"Quiz for {title}",
                    "questions": [
                        {
                            "question_text": f"This quiz tests your knowledge of {title}.",
                            "question_type": "MCQ",
                            "options": {
                                "A": "True",
                                "B": "False"
                            },
                            "correct_answer": "A",
                            "explanation": "This is a placeholder question."
                        }
                    ]
                }
                
        except Exception as e:
            print(f"Error generating quiz: {e}")
            # Return a basic fallback quiz
            return {
                "quiz_title": f"Quiz for {title}",
                "questions": [
                    {
                        "question_text": f"This quiz tests your knowledge of {title}.",
                        "question_type": "MCQ",
                        "options": {
                            "A": "True",
                            "B": "False"
                        },
                        "correct_answer": "A",
                        "explanation": "This is a placeholder question."
                    }
                ]
            }
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from text response that might contain additional commentary
        
        Args:
            text: Text response from AI that might contain JSON
            
        Returns:
            Extracted JSON string or empty string if not found
        """
        if not text:
            return ""
            
        # Try to find JSON array or object pattern
        json_match = re.search(r'(\[\s*{.*}\s*\]|{.*})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
        # Look for markdown code blocks with JSON
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
            
        # If no clear pattern, return the whole text (it might be pure JSON)
        return text
    
    def _try_fix_json(self, text: str) -> str:
        """
        Try to fix common JSON formatting issues
        
        Args:
            text: Text that might contain malformed JSON
            
        Returns:
            Fixed JSON string or empty string if not fixable
        """
        if not text:
            return ""
            
        # Extract what looks like JSON
        json_text = self._extract_json_from_text(text)
        
        # Fix common issues
        try:
            # Fix single quotes to double quotes (but not inside already quoted strings)
            # This is complex, so let's do a simpler version
            fixed = re.sub(r"(?<!\\)'([^']*?)(?<!\\)'(?=[,\s\]\}:]|$)", r'"\1"', json_text)
            
            # Fix unquoted keys
            fixed = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', fixed)
            
            # Fix trailing commas in objects/arrays
            fixed = re.sub(r',\s*([\]\}])', r'\1', fixed)
            
            # Test if it's valid
            json.loads(fixed)
            return fixed
        except:
            # Try a second approach - just look for the JSON markers
            try:
                start_idx = text.find('[')
                if start_idx == -1:
                    start_idx = text.find('{')
                
                if start_idx != -1:
                    # Find matching close bracket
                    stack = []
                    for i in range(start_idx, len(text)):
                        if text[i] in '[{':
                            stack.append(text[i])
                        elif text[i] == ']' and stack and stack[-1] == '[':
                            stack.pop()
                            if not stack:  # Found matching brackets
                                extracted = text[start_idx:i+1]
                                # Test if valid
                                json.loads(extracted)
                                return extracted
                        elif text[i] == '}' and stack and stack[-1] == '{':
                            stack.pop()
                            if not stack:  # Found matching brackets
                                extracted = text[start_idx:i+1]
                                # Test if valid
                                json.loads(extracted)
                                return extracted
            except:
                pass
                
            return ""
    
    def _select_diverse_questions(self, questions: List[Dict[str, Any]], num_questions: int = 15) -> List[Dict[str, Any]]:
        """
        Select a diverse set of questions from a larger pool
        
        Args:
            questions: List of all generated questions
            num_questions: Number of questions to select
            
        Returns:
            Diverse subset of questions
        """
        if len(questions) <= num_questions:
            return questions
        
        # Ensure we select a mix of question types
        mcq_questions = [q for q in questions if q.get("question_type") == "MCQ"]
        tf_questions = [q for q in questions if q.get("question_type") == "TRUE_FALSE"]
        
        # Calculate proportions for selection
        total_mcq = len(mcq_questions)
        total_tf = len(tf_questions)
        total = total_mcq + total_tf
        
        # Determine how many of each type to include
        if total == 0:
            return questions[:num_questions]  # Fallback if no type information
        
        mcq_proportion = total_mcq / total
        tf_proportion = total_tf / total
        
        num_mcq = math.ceil(num_questions * mcq_proportion)
        num_tf = num_questions - num_mcq
        
        # Select questions
        selected_mcq = mcq_questions[:num_mcq] if total_mcq > 0 else []
        selected_tf = tf_questions[:num_tf] if total_tf > 0 else []
        
        # Combine and ensure we don't exceed the desired number
        selected = selected_mcq + selected_tf
        return selected[:num_questions]
    
    def _generate_ai_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate AI response using available model
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            
        Returns:
            AI-generated response text
        """
        try:
            if self.llm:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                return response.content
            elif self.gemini_model:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(full_prompt)
                return response.text
            elif self.openrouter_model:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.openrouter_model.invoke(messages)
                return response.content
            else:
                return "No AI model available for content generation."
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return ""
    
    def _save_content_to_db(self, document: Document, lessons: List[Dict[str, Any]], quiz_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Save generated content to the database
        
        Args:
            document: Document model
            lessons: Generated lessons
            quiz_data: Generated quiz data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Save the lesson content to document metadata
            if not document.doc_metadata:
                document.doc_metadata = {}
            
            document.doc_metadata["lessons"] = lessons
            document.doc_metadata["last_updated"] = datetime.now().isoformat()
            
            # Create quiz record
            quiz = Quiz(
                title=quiz_data["quiz_title"],
                document_id=document.id
            )
            self.db.add(quiz)
            self.db.flush()  # Flush to get the quiz ID
            
            # Create question records
            for q_data in quiz_data["questions"]:
                    # Try to find a matching chunk based on the source text if provided
                    source_chunk_id = None
                    if "source_text" in q_data and q_data["source_text"]:
                        source_text = q_data["source_text"]
                        # Look for chunks containing this text
                        try:
                            matching_chunk = self.db.query(Chunk).filter(
                                Chunk.content.contains(source_text[:50])  # Use first part of source text
                            ).filter(
                                Chunk.chunk_metadata.op('->>')('document_id') == str(document.id)
                            ).first()
                            
                            if matching_chunk:
                                source_chunk_id = matching_chunk.id
                        except Exception as e:
                            print(f"Error finding matching chunk: {e}")
                            
                    question = Question(
                        quiz_id=quiz.id,
                        question_text=q_data["question_text"],
                        question_type=q_data["question_type"],
                        options=q_data["options"],
                        correct_answer=q_data["correct_answer"],
                        explanation=q_data["explanation"],
                        source_chunk_id=source_chunk_id
                    )
                    self.db.add(question)
            
            # Commit changes
            self.db.commit()
            
            return True, f"Successfully generated content for document: {document.title}"
            
        except Exception as e:
            self.db.rollback()
            return False, f"Error saving content to database: {str(e)}"
