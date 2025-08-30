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

from models.models import Document, Quiz, Question, TrainingRecord, Chunk, LearningModule, LearningSession, SessionContent
from utils.rag_engine import RAGEngine

# Define schema for AI responses
CHUNK_ANALYSIS_SCHEMA = """
{
  "is_food_beverage_procedure": boolean,  // REQUIRED: true only if content describes physical food/beverage work processes
  "procedure_type": string,               // "equipment_operation", "food_preparation", "cleaning_process", "safety_procedure", "not_applicable"
  "has_learning_content": boolean,       // Whether this chunk contains material suitable for learning content
  "has_quiz_content": boolean,           // Whether this chunk contains material suitable for quiz questions
  "learning_lessons": [                   // Array of lessons if has_learning_content is true, empty array otherwise
    {
      "title": string,
      "summary": string,
      "key_points": [string],
      "best_practices": [string],
      "category": "physical_work_process"
    }
  ],
  "quiz_questions": [                     // Array of questions if has_quiz_content is true, empty array otherwise
    {
      "question_text": string,
      "question_type": string,             // "MCQ" or "TRUE_FALSE"
      "options": {"A": string, "B": string, "C": string, "D": string},
      "correct_answer": string,
      "explanation": string,
      "category": "physical_work_process"
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
                    max_tokens=2048,  # Reduced token limit to save credits
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
            
            # Add batch processing to handle credits more efficiently
            batch_size = 50  # Process in smaller batches to monitor credits
            total_batches = math.ceil(len(valid_chunks) / batch_size)
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(valid_chunks))
                batch_chunks = valid_chunks[start_idx:end_idx]
                
                print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)")
                
                try:
                    for i, chunk in enumerate(batch_chunks):
                        chunk_idx = start_idx + i + 1
                        print(f"Processing chunk {chunk_idx}/{len(valid_chunks)}")
                        chunk_content = chunk.get("content", "")
                        chunk_metadata = chunk.get("metadata", {})
                        
                        # For each chunk, determine if it contains material for learning content and/or quiz
                        chunk_lessons, chunk_questions = self._process_individual_chunk(
                            document.title, 
                            chunk_content, 
                            chunk_metadata,
                            chunk_id=chunk.get("chunk_id")
                        )
                        
                        # Save to database immediately if content was generated
                        if chunk_lessons or chunk_questions:
                            print(f"  - Found {len(chunk_lessons)} lessons and {len(chunk_questions)} questions - saving to DB")
                            
                            # Save this chunk's content to database
                            try:
                                self._save_chunk_content_to_db(document, chunk_lessons, chunk_questions)
                                all_lessons.extend(chunk_lessons)
                                all_questions.extend(chunk_questions)
                            except Exception as save_error:
                                print(f"  - Error saving chunk content: {save_error}")
                        else:
                            print(f"  - No F&B procedure content found in chunk {chunk_idx}")
                        
                except Exception as e:
                    if "credits" in str(e).lower() or "payment" in str(e).lower():
                        print(f"⚠️  Credits exhausted at batch {batch_num + 1}. Processed {start_idx} chunks.")
                        print(f"Generated content so far: {len(all_lessons)} lessons, {len(all_questions)} questions")
                        if all_lessons or all_questions:
                            print("Proceeding with partial content generation...")
                            break
                        else:
                            return False, "OpenRouter credits exhausted before generating any content"
                    else:
                        raise e
            
            # Create consolidated results (even if saved incrementally)
            final_message = f"Processed {len(valid_chunks)} chunks. Generated {len(all_lessons)} lessons and {len(all_questions)} questions for food & beverage procedures."
            
            # Create structured learning modules and sessions if we have content
            if all_lessons or all_questions:
                try:
                    self._create_learning_modules_and_sessions(document, all_lessons, all_questions)
                    return True, final_message
                except Exception as e:
                    print(f"Error creating learning modules: {e}")
                    return True, f"{final_message} Warning: Could not create learning modules."
            else:
                return True, "No food & beverage procedures found in document. No content generated."
            
            return True, final_message
            
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
            
            # Limit chunk content to save tokens
            max_chunk_length = 3000
            if len(chunk_content) > max_chunk_length:
                chunk_content = chunk_content[:max_chunk_length] + "..."
                print(f"Truncated chunk to {max_chunk_length} characters to save tokens")
            
            # Create the analysis prompt - SHORT AND FOCUSED
            system_prompt = f"""
You analyze food & beverage work procedures. Respond ONLY with JSON.

ONLY generate content for physical food/beverage processes like:
- Equipment operation (fryers, ovens, mixers, dishwashers)
- Food preparation steps (cutting, cooking, mixing, serving)
- Cleaning procedures (sanitizing equipment, surfaces)
- Safety procedures in food handling

SKIP: administrative content, company policies, general information.

JSON format:
{CHUNK_ANALYSIS_SCHEMA}

Generate 1-2 lessons max, 2-4 questions max.
"""

            user_prompt = f"""
Document: {document_title}
Content:
{chunk_content}

Analyze for food/beverage physical procedures only. Return JSON.
"""

            # Get AI response with error handling for credits
            response = self._generate_ai_response(system_prompt, user_prompt)
            
            if not response:
                print(f"No response from AI for chunk {chunk_id} (possible credits issue)")
                # If we're running out of credits, stop processing more chunks
                if "credits" in str(response).lower() or "payment" in str(response).lower():
                    print("⚠️  OpenRouter credits exhausted. Stopping content generation.")
                    raise Exception("OpenRouter credits exhausted")
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
                is_food_beverage_procedure = analysis.get('is_food_beverage_procedure', False)
                procedure_type = analysis.get('procedure_type', 'not_applicable')
                
                # Log analysis
                print(f"Food/beverage procedure: {is_food_beverage_procedure}, Type: {procedure_type}")
                
                # Filter: Only include content if it's a food & beverage procedure
                if not is_food_beverage_procedure or procedure_type == 'not_applicable':
                    print(f"Skipping content - not a food/beverage procedure")
                    return [], []  # Return empty lists for non-F&B procedures
                
                print(f"Generating content for F&B procedure: {procedure_type}")
                
                # Add chunk metadata to lessons
                for lesson in lessons:
                    lesson['source_chunk_id'] = chunk_id
                    lesson['source_metadata'] = chunk_metadata
                    lesson['category'] = 'physical_work_process'  # Set as physical work process
                
                # Add chunk metadata to questions
                for question in questions:
                    question['source_chunk_id'] = chunk_id
                    question['source_metadata'] = chunk_metadata
                    question['category'] = 'physical_work_process'  # Set as physical work process
                    # Ensure source_text is included
                    if 'source_text' not in question:
                        question['source_text'] = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
                
                print(f"Chunk analysis: {len(lessons)} lessons, {len(questions)} questions for {procedure_type}")
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
                            lesson['category'] = 'physical_work_process'
                        
                        for question in questions:
                            question['source_chunk_id'] = chunk_id
                            question['source_metadata'] = chunk_metadata
                            question['category'] = 'physical_work_process'
                            if 'source_text' not in question:
                                question['source_text'] = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
                        
                        # Apply F&B procedure filtering again
                        is_food_beverage_procedure = analysis.get('is_food_beverage_procedure', False)
                        if not is_food_beverage_procedure:
                            print(f"Skipping fixed content - not a food/beverage procedure")
                            return [], []
                        
                        print(f"Fixed JSON parsing - Found {len(lessons)} lessons, {len(questions)} questions for F&B procedure")
                        return lessons, questions
                    except Exception as fix_error:
                        print(f"Error even after fixing JSON for chunk {chunk_id}: {fix_error}")
                
                return [], []
                
        except Exception as e:
            print(f"Error processing individual chunk {chunk_id}: {e}")
            
            # Check if it's a credits/payment error
            if "credits" in str(e).lower() or "payment" in str(e).lower() or "402" in str(e):
                print("⚠️  OpenRouter API credits exhausted!")
                print("Solutions:")
                print("1. Add credits at https://openrouter.ai/settings/credits")
                print("2. Use a different AI provider (OpenAI/Gemini)")
                print("3. Process fewer chunks at a time")
                # Re-raise to stop processing
                raise Exception("OpenRouter credits exhausted") from e
            
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
            # Determine the primary category for this quiz based on questions
            quiz_category = self._determine_primary_category(quiz_data["questions"])
            
            quiz = Quiz(
                title=quiz_data["quiz_title"],
                document_id=document.id,
                category=quiz_category
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
                        source_chunk_id=source_chunk_id,
                        category=q_data.get("category", "other")
                    )
                    self.db.add(question)
            
            # Commit changes
            self.db.commit()
            
            return True, f"Successfully generated content for document: {document.title}"
            
        except Exception as e:
            self.db.rollback()
            return False, f"Error saving content to database: {str(e)}"
    
    def _save_chunk_content_to_db(self, document: Document, lessons: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> bool:
        """
        Save individual chunk's content to database immediately
        
        Args:
            document: Document model
            lessons: Generated lessons from this chunk
            questions: Generated quiz questions from this chunk
            
        Returns:
            Success status
        """
        try:
            # Create or get existing quiz for this document
            existing_quiz = self.db.query(Quiz).filter(Quiz.document_id == document.id).first()
            
            if not existing_quiz:
                # Create new quiz
                quiz = Quiz(
                    title=f"F&B Procedures - {document.title}",
                    document_id=document.id,
                    category="physical_work_process"
                )
                self.db.add(quiz)
                self.db.flush()  # Get the quiz ID
            else:
                quiz = existing_quiz
            
            # Add questions to quiz
            for q_data in questions:
                question = Question(
                    quiz_id=quiz.id,
                    question_text=q_data["question_text"],
                    question_type=q_data["question_type"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    explanation=q_data["explanation"],
                    category="physical_work_process"
                )
                self.db.add(question)
            
            # Save lessons to document metadata
            if not document.doc_metadata:
                document.doc_metadata = {}
            
            if "lessons" not in document.doc_metadata:
                document.doc_metadata["lessons"] = []
            
            # Add new lessons to existing ones
            document.doc_metadata["lessons"].extend(lessons)
            document.doc_metadata["last_updated"] = datetime.now().isoformat()
            
            # Mark as modified for SQLAlchemy
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(document, "doc_metadata")
            
            # Commit changes
            self.db.commit()
            
            return True
            
        except Exception as e:
            print(f"Error saving chunk content to database: {e}")
            self.db.rollback()
            return False
    
    def _determine_content_category(self, content: Dict[str, Any]) -> str:
        """
        Determine the primary category for content (lessons + questions)
        
        Args:
            content: Dictionary with 'lessons' and 'questions' keys
            
        Returns:
            Primary category string
        """
        all_items = content.get('lessons', []) + content.get('questions', [])
        
        if not all_items:
            return "other"
            
        # Count categories
        category_counts = {}
        for item in all_items:
            category = item.get("category", "other")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Return the most common category
        if category_counts:
            return max(category_counts, key=category_counts.get)
        return "other"
    
    def _determine_primary_category(self, questions: List[Dict[str, Any]]) -> str:
        """
        Determine the primary category for a quiz based on its questions
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Primary category string
        """
        if not questions:
            return "other"
            
        # Count categories
        category_counts = {}
        for question in questions:
            category = question.get("category", "other")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Return the most common category
        if category_counts:
            return max(category_counts, key=category_counts.get)
        return "other"
    
    def _create_learning_modules_and_sessions(
        self, 
        document: Document, 
        lessons: List[Dict[str, Any]], 
        questions: List[Dict[str, Any]]
    ):
        """
        Create structured learning modules and sessions from generated content
        
        Args:
            document: Document model
            lessons: Generated lessons
            questions: Generated quiz questions
        """
        try:
            print(f"Creating learning modules and sessions for document {document.id}")
            
            # Group lessons and questions by topics/themes
            grouped_content = self._group_content_by_topics(lessons, questions)
            
            # Create learning modules
            for i, (topic, content) in enumerate(grouped_content.items()):
                # Determine the primary category for this module
                module_category = self._determine_content_category(content)
                
                # Create module
                module = LearningModule(
                    title=topic,
                    description=f"Learn about {topic.lower()} from {document.title}",
                    document_id=document.id,
                    module_order=i + 1,
                    estimated_duration=self._estimate_module_duration(content),
                    difficulty_level=self._determine_difficulty_level(content),
                    learning_objectives=self._extract_learning_objectives(content['lessons']),
                    category=module_category
                )
                
                self.db.add(module)
                self.db.flush()  # Get the module ID
                
                # Create sessions for this module
                self._create_sessions_for_module(module, content)
                
            self.db.commit()
            print(f"Successfully created {len(grouped_content)} learning modules")
            
        except Exception as e:
            print(f"Error creating learning modules and sessions: {e}")
            traceback.print_exc()
            self.db.rollback()
    
    def _group_content_by_topics(self, lessons: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group lessons and questions by common topics/themes
        
        Args:
            lessons: List of lesson dictionaries
            questions: List of question dictionaries
            
        Returns:
            Dictionary mapping topic names to content
        """
        grouped = {}
        
        # If we have few lessons/questions, create a single module
        if len(lessons) <= 3 and len(questions) <= 20:
            topic_name = lessons[0].get('title', 'Learning Module') if lessons else 'Quiz Module'
            # Clean up the topic name
            topic_name = topic_name.replace('Learning content for ', '').replace('Quiz for ', '')
            
            grouped[topic_name] = {
                'lessons': lessons,
                'questions': questions
            }
            return grouped
        
        # For larger content, group by keywords and themes
        # Simple grouping based on lesson titles and key terms
        topic_keywords = {}
        
        # Extract keywords from lessons
        for lesson in lessons:
            title = lesson.get('title', '')
            key_points = lesson.get('key_points', [])
            
            # Extract key terms from title and key points
            terms = self._extract_key_terms_from_lesson(title, key_points)
            
            # Find best matching topic or create new one
            best_topic = self._find_best_topic_match(terms, topic_keywords)
            
            if best_topic:
                if best_topic not in grouped:
                    grouped[best_topic] = {'lessons': [], 'questions': []}
                grouped[best_topic]['lessons'].append(lesson)
            else:
                # Create new topic
                topic_name = self._generate_topic_name(title, terms)
                if topic_name not in grouped:
                    grouped[topic_name] = {'lessons': [], 'questions': []}
                    topic_keywords[topic_name] = terms
                grouped[topic_name]['lessons'].append(lesson)
        
        # Distribute questions among topics based on content similarity
        for question in questions:
            question_text = question.get('question_text', '')
            source_text = question.get('source_text', '')
            
            # Find best topic for this question
            best_topic = self._find_best_topic_for_question(
                question_text + ' ' + source_text, 
                grouped
            )
            
            if best_topic:
                grouped[best_topic]['questions'].append(question)
            else:
                # Add to first available topic or create general topic
                if grouped:
                    first_topic = list(grouped.keys())[0]
                    grouped[first_topic]['questions'].append(question)
                else:
                    grouped['General Knowledge'] = {'lessons': [], 'questions': [question]}
        
        # Ensure no empty modules and reasonable distribution
        grouped = self._balance_content_distribution(grouped)
        
        return grouped
    
    def _extract_key_terms_from_lesson(self, title: str, key_points: List[str]) -> List[str]:
        """Extract key terms from lesson title and key points."""
        import re
        
        text = title + ' ' + ' '.join(key_points)
        
        # Extract meaningful terms (2+ characters, not common words)
        terms = re.findall(r'\b[A-Za-z]{2,}\b', text.lower())
        
        # Filter out common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say', 'each', 'which', 'their', 'time', 'will', 'about', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'what', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'when', 'come', 'its', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'see', 'own', 'other', 'after', 'first', 'well', 'year', 'where', 'much', 'should', 'people', 'how', 'before', 'take', 'good', 'through', 'could', 'great', 'little', 'world', 'still', 'back', 'here', 'because', 'even', 'right', 'want', 'too', 'any', 'most', 'old', 'need', 'said', 'such', 'being', 'have', 'that', 'this', 'with', 'they', 'from', 'there', 'been', 'were', 'know', 'just', 'down', 'last', 'long', 'might', 'came', 'every', 'made', 'look', 'find', 'between', 'never', 'again', 'things', 'going', 'does', 'different', 'away', 'something', 'without', 'another', 'around', 'during', 'really', 'important', 'almost', 'those', 'both', 'until', 'however', 'while', 'part'}
        
        meaningful_terms = [term for term in terms if term not in common_words and len(term) > 2]
        
        # Return top 5 most frequent terms
        from collections import Counter
        term_counts = Counter(meaningful_terms)
        return [term for term, count in term_counts.most_common(5)]
    
    def _find_best_topic_match(self, terms: List[str], existing_topics: Dict[str, List[str]]) -> Optional[str]:
        """Find the best matching existing topic for the given terms."""
        if not existing_topics or not terms:
            return None
        
        best_match = None
        best_score = 0
        
        for topic, topic_terms in existing_topics.items():
            # Calculate overlap score
            overlap = len(set(terms) & set(topic_terms))
            total_terms = len(set(terms) | set(topic_terms))
            score = overlap / total_terms if total_terms > 0 else 0
            
            if score > best_score and score > 0.3:  # Minimum similarity threshold
                best_score = score
                best_match = topic
        
        return best_match
    
    def _generate_topic_name(self, title: str, terms: List[str]) -> str:
        """Generate a topic name from lesson title and terms."""
        # Clean up title
        clean_title = title.replace('Learning content for ', '').replace('Quiz for ', '')
        
        # If title is meaningful, use it
        if len(clean_title) > 5 and not clean_title.startswith('Document'):
            return clean_title[:50]  # Limit length
        
        # Otherwise, create from key terms
        if terms:
            return ' '.join(terms[:3]).title()
        
        return 'General Topics'
    
    def _find_best_topic_for_question(self, question_content: str, grouped: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Find the best topic for a question based on content similarity."""
        if not grouped:
            return None
        
        question_terms = self._extract_key_terms(question_content.lower(), max_terms=10)
        
        best_topic = None
        best_score = 0
        
        for topic, content in grouped.items():
            # Get terms from lessons in this topic
            topic_terms = []
            for lesson in content['lessons']:
                lesson_terms = self._extract_key_terms_from_lesson(
                    lesson.get('title', ''), 
                    lesson.get('key_points', [])
                )
                topic_terms.extend(lesson_terms)
            
            # Calculate similarity
            overlap = len(set(question_terms) & set(topic_terms))
            total_terms = len(set(question_terms) | set(topic_terms))
            score = overlap / total_terms if total_terms > 0 else 0
            
            if score > best_score:
                best_score = score
                best_topic = topic
        
        return best_topic if best_score > 0.1 else None
    
    def _balance_content_distribution(self, grouped: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Balance content distribution across topics."""
        if not grouped:
            return grouped
        
        # Remove empty topics
        filtered = {}
        for topic, content in grouped.items():
            if content['lessons'] or content['questions']:
                filtered[topic] = content
        
        # If too many small topics, merge some
        if len(filtered) > 10:
            # Merge topics with very few items
            merged = {}
            general_content = {'lessons': [], 'questions': []}
            
            for topic, content in filtered.items():
                total_items = len(content['lessons']) + len(content['questions'])
                if total_items >= 3:  # Keep substantial topics
                    merged[topic] = content
                else:  # Merge small topics
                    general_content['lessons'].extend(content['lessons'])
                    general_content['questions'].extend(content['questions'])
            
            if general_content['lessons'] or general_content['questions']:
                merged['Additional Topics'] = general_content
            
            filtered = merged
        
        return filtered or {'Learning Module': {'lessons': [], 'questions': []}}
    
    def _create_sessions_for_module(self, module: LearningModule, content: Dict[str, Any]):
        """Create learning sessions for a module."""
        lessons = content.get('lessons', [])
        questions = content.get('questions', [])
        
        session_order = 1
        
        # Create lesson sessions (if we have lessons)
        if lessons:
            # Group lessons into sessions (max 3 lessons per session)
            lesson_batches = [lessons[i:i+3] for i in range(0, len(lessons), 3)]
            
            for i, lesson_batch in enumerate(lesson_batches):
                session_title = f"{module.title} - Lessons {i+1}"
                if len(lesson_batches) == 1:
                    session_title = f"{module.title} - Learning Content"
                
                session = LearningSession(
                    title=session_title,
                    description=f"Learn the key concepts and procedures for {module.title.lower()}",
                    module_id=module.id,
                    session_order=session_order,
                    session_type="lesson",
                    estimated_duration=len(lesson_batch) * 5,  # 5 min per lesson
                    is_required=True,
                    category=module.category
                )
                
                self.db.add(session)
                self.db.flush()
                
                # Add lesson content to session
                for j, lesson in enumerate(lesson_batch):
                    content_item = SessionContent(
                        session_id=session.id,
                        content_type="lesson",
                        content_order=j + 1,
                        lesson_data=lesson,
                        category=lesson.get('category', module.category)
                    )
                    self.db.add(content_item)
                
                session_order += 1
        
        # Create quiz sessions (if we have questions)
        if questions:
            # Group questions into sessions (5-10 questions per session)
            questions_per_session = min(10, max(5, len(questions) // 3)) if len(questions) > 15 else len(questions)
            question_batches = [questions[i:i+questions_per_session] for i in range(0, len(questions), questions_per_session)]
            
            for i, question_batch in enumerate(question_batches):
                session_title = f"{module.title} - Quiz {i+1}"
                if len(question_batches) == 1:
                    session_title = f"{module.title} - Assessment"
                
                session = LearningSession(
                    title=session_title,
                    description=f"Test your knowledge of {module.title.lower()}",
                    module_id=module.id,
                    session_order=session_order,
                    session_type="quiz",
                    estimated_duration=len(question_batch) * 2,  # 2 min per question
                    passing_score=70,
                    max_attempts=3,
                    is_required=True,
                    category=module.category
                )
                
                self.db.add(session)
                self.db.flush()
                
                # Add questions to session
                for j, question_data in enumerate(question_batch):
                    # First, create the question in the database if not already exists
                    question = Question(
                        quiz_id=None,  # We'll link via session content instead
                        question_text=question_data['question_text'],
                        question_type=question_data['question_type'],
                        options=question_data['options'],
                        correct_answer=question_data['correct_answer'],
                        explanation=question_data['explanation'],
                        category=question_data.get('category', module.category)
                    )
                    self.db.add(question)
                    self.db.flush()
                    
                    # Add question to session content
                    content_item = SessionContent(
                        session_id=session.id,
                        content_type="quiz_question",
                        content_order=j + 1,
                        question_id=question.id,
                        category=question_data.get('category', module.category)
                    )
                    self.db.add(content_item)
                
                session_order += 1
    
    def _estimate_module_duration(self, content: Dict[str, Any]) -> int:
        """Estimate module duration in minutes."""
        lessons = content.get('lessons', [])
        questions = content.get('questions', [])
        
        # 5 minutes per lesson + 2 minutes per question
        return len(lessons) * 5 + len(questions) * 2
    
    def _determine_difficulty_level(self, content: Dict[str, Any]) -> str:
        """Determine difficulty level based on content complexity."""
        lessons = content.get('lessons', [])
        questions = content.get('questions', [])
        
        # Simple heuristic based on content amount and complexity
        total_content = len(lessons) + len(questions)
        
        if total_content <= 10:
            return "Beginner"
        elif total_content <= 25:
            return "Intermediate"
        else:
            return "Advanced"
    
    def _extract_learning_objectives(self, lessons: List[Dict[str, Any]]) -> List[str]:
        """Extract learning objectives from lessons."""
        objectives = []
        
        for lesson in lessons[:3]:  # Limit to first 3 lessons
            title = lesson.get('title', '')
            key_points = lesson.get('key_points', [])
            
            # Create objective from title
            if title and not title.startswith('Learning content'):
                clean_title = title.replace('Learning content for ', '')
                objectives.append(f"Understand {clean_title.lower()}")
            
            # Add key points as objectives
            for point in key_points[:2]:  # Max 2 points per lesson
                if point and len(point) > 10:
                    objectives.append(f"Learn {point.lower()}")
        
        return objectives[:5]  # Maximum 5 objectives
