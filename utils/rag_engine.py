import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
import re
import requests
import traceback

# Handle optional imports gracefully
LANGCHAIN_AVAILABLE = False
GEMINI_AVAILABLE = False

# Try to import LangChain libraries
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain libraries not available: {e}")
    # Define stub classes to prevent errors
    class SystemMessage:
        def __init__(self, content): self.content = content
    class HumanMessage:
        def __init__(self, content): self.content = content
    class ChatOpenAI:
        def __init__(self, **kwargs): pass
        def invoke(self, messages): return type('obj', (object,), {'content': 'LangChain not available'})
    class OpenAIEmbeddings:
        def __init__(self, **kwargs): pass
        def embed_query(self, text): return [0.0] * 384

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Google Generative AI library not available: {e}")
    # Create stub module
    class GeminiStub:
        def GenerativeModel(self, *args): return self
        def generate_content(self, text): return type('obj', (object,), {'text': 'Gemini not available'})
        def configure(self, **kwargs): pass
    genai = GeminiStub()

from models.models import Document, Chunk

class RAGEngine:
    def __init__(self, db: Session):
        self.db = db
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Initialize AI models with fallback options
        self.llm = None
        self.embeddings = None
        self.gemini_model = None
        self.openrouter_model = None
        
        # Try OpenAI first if LangChain is available
        if LANGCHAIN_AVAILABLE and self.openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=self.openai_api_key
                )
                self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
                print("Using OpenAI for LLM and embeddings")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
                traceback.print_exc()
        
        # Try Gemini as fallback if available
        if not self.llm and GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                print("Using Gemini for LLM")
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                traceback.print_exc()
        
        # Try OpenRouter as fallback if LangChain is available
        if not self.llm and not self.gemini_model and LANGCHAIN_AVAILABLE and self.openrouter_api_key:
            try:
                self.openrouter_model = ChatOpenAI(
                    model="openai/gpt-3.5-turbo",
                    temperature=0.1,
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=self.openrouter_api_key
                )
                print("Using OpenRouter for LLM")
            except Exception as e:
                print(f"Error initializing OpenRouter: {e}")
                traceback.print_exc()
        
        # Log warning if no AI models are available
        if not LANGCHAIN_AVAILABLE and not GEMINI_AVAILABLE:
            print("WARNING: No AI libraries available. Using fallback responses.")
        elif not self.llm and not self.gemini_model and not self.openrouter_model:
            print("WARNING: No AI models could be initialized. Using fallback responses.")
        
    def query(
        self, 
        query: str, 
        document_ids: Optional[List[int]] = None,
        procedure_mode: bool = False,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Query the documents using RAG.
        
        Args:
            query: The user's query
            document_ids: Optional list of document IDs to restrict the search
            procedure_mode: Whether to return step-by-step procedure
            language: Language code for the response
            
        Returns:
            Dict containing the answer and source citations
        """
        # Get relevant documents
        if document_ids:
            documents = self.db.query(Document).filter(Document.id.in_(document_ids)).all()
        else:
            documents = self.db.query(Document).all()
        
        if not documents:
            return {
                "answer": "I don't have any documents to search through.",
                "sources": []
            }
        
        # Process each document using cached embeddings when available
        all_chunks = []
        for doc in documents:
            print(f"Processing document ID: {doc.id}")
            
            # Try to load cached data first
            cached_chunks, cached_embeddings = self._load_cached_embeddings(doc.id)
            
            if cached_chunks and cached_embeddings and len(cached_chunks) == len(cached_embeddings):
                print(f"Using cached embeddings for document {doc.id}")
                # Use cached data
                for chunk_content, chunk_embedding in zip(cached_chunks, cached_embeddings):
                    if self.embeddings:
                        score = self._calculate_embedding_similarity(query, chunk_content, cached_embedding=chunk_embedding)
                    else:
                        score = self._calculate_similarity(query, chunk_content)
                    
                    all_chunks.append({
                        "document": doc,
                        "content": chunk_content,
                        "metadata": {"document_id": doc.id, "section": "cached_chunk"},
                        "score": score
                    })
            else:
                print(f"Falling back to database chunks for document {doc.id}")
                # Fallback to database chunks
                chunks = []
                
                # Method 1: JSON query
                try:
                    chunks = self.db.query(Chunk).filter(
                        Chunk.chunk_metadata.op('->>')('document_id') == str(doc.id)
                    ).all()
                    print(f"Found {len(chunks)} chunks using JSON query")
                except Exception as e:
                    print(f"JSON query failed: {e}")
                
                # Method 2: If no chunks found, try direct relationship
                if not chunks:
                    try:
                        chunks = doc.chunks
                        print(f"Found {len(chunks)} chunks using direct relationship")
                    except Exception as e:
                        print(f"Direct relationship failed: {e}")
                
                # Method 3: If still no chunks, try simple text search
                if not chunks:
                    try:
                        chunks = self.db.query(Chunk).filter(
                            Chunk.content.contains(doc.title)
                        ).all()
                        print(f"Found {len(chunks)} chunks using text search")
                    except Exception as e:
                        print(f"Text search failed: {e}")
                
                # Method 4: Get all chunks if still none found
                if not chunks:
                    try:
                        chunks = self.db.query(Chunk).all()
                        print(f"Found {len(chunks)} total chunks in database")
                    except Exception as e:
                        print(f"Getting all chunks failed: {e}")
                
                for chunk in chunks:
                    # Use proper embeddings if available, otherwise fallback to simple similarity
                    if self.embeddings:
                        score = self._calculate_embedding_similarity(query, chunk.content)
                    else:
                        score = self._calculate_similarity(query, chunk.content)
                    
                    all_chunks.append({
                        "document": doc,
                        "content": chunk.content,
                        "metadata": chunk.chunk_metadata,
                        "score": score
                    })
        
        if not all_chunks:
            return {
                "answer": "No searchable content found in the requested documents.",
                "sources": []
            }
        
        # Sort by relevance score
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top results
        top_results = all_chunks[:5]
        
        # Prepare context for the response
        context = "\n\n".join([
            f"Document: {r['document'].title} (v{r['document'].version})\n"
            f"Section: {r['metadata'].get('section', 'N/A')}\n"
            f"Content: {r['content']}"
            for r in top_results
        ])
        
        # Prepare sources for citation
        sources = [
            {
                "document_title": r["document"].title,
                "document_version": r["document"].version,
                "section": r["metadata"].get("section", "N/A"),
                "page": r["metadata"].get("page_number", "N/A"),
            }
            for r in top_results
        ]
        
        # Generate response based on mode using available AI model
        if self.llm:
            if procedure_mode:
                answer = self._generate_ai_procedure_response(query, context, sources, "openai")
            else:
                answer = self._generate_ai_regular_response(query, context, sources, "openai")
        elif self.gemini_model:
            if procedure_mode:
                answer = self._generate_ai_procedure_response(query, context, sources, "gemini")
            else:
                answer = self._generate_ai_regular_response(query, context, sources, "gemini")
        elif self.openrouter_model:
            if procedure_mode:
                answer = self._generate_ai_procedure_response(query, context, sources, "openrouter")
            else:
                answer = self._generate_ai_regular_response(query, context, sources, "openrouter")
        else:
            # Fallback to simplified responses if no AI model available
            if procedure_mode:
                answer = self._generate_procedure_response(query, context, sources)
            else:
                answer = self._generate_regular_response(query, context, sources)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def _load_cached_embeddings(self, doc_id: int) -> Tuple[List[str], List[List[float]]]:
        """Load cached embeddings for a document."""
        index_dir = os.path.join("indexes", str(doc_id))
        chunks = []
        embeddings = []
        
        try:
            # Load chunks from index.json
            with open(os.path.join(index_dir, "index.json"), "r") as f:
                index_data = json.load(f)
                chunks = index_data["chunks"]
            
            # Load cached embeddings from pickle file
            embeddings_file = os.path.join(index_dir, "embeddings.pkl")
            if os.path.exists(embeddings_file):
                with open(embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
                print(f"Loaded cached embeddings for document {doc_id}")
        except Exception as e:
            print(f"Error loading cached data for document {doc_id}: {e}")
        
        return chunks, embeddings

    def _calculate_embedding_similarity(self, query: str, content: str, cached_embedding: Optional[List[float]] = None) -> float:
        """Calculate similarity using embeddings."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Use cached embedding if available, otherwise generate new one
            if cached_embedding is not None:
                content_embedding = cached_embedding
            else:
                content_embedding = self.embeddings.embed_query(content)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_embedding, content_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_content = np.linalg.norm(content_embedding)
            
            if norm_query == 0 or norm_content == 0:
                return 0.0
            
            return dot_product / (norm_query * norm_content)
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return self._calculate_similarity(query, content)
    
    def _calculate_similarity(self, query: str, content: str) -> float:
        """Calculate similarity between query and content (simplified version)."""
        # Convert to lowercase for comparison
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Simple word overlap similarity
        query_words = set(re.findall(r'\w+', query_lower))
        content_words = set(re.findall(r'\w+', content_lower))
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_ai_procedure_response(self, query: str, context: str, sources: List[Dict], model_type: str = "openai") -> str:
        """Generate a step-by-step procedure response using AI."""
        try:
            system_prompt = """You are an expert Food & Beverage training assistant. Your role is to provide clear, step-by-step procedures based on SOP documents and training materials.

When providing procedures:
1. Extract specific steps from the provided context
2. Organize them in logical order
3. Include safety warnings and important notes
4. Be concise but comprehensive
5. Always cite your sources

Format your response with clear numbered steps and include safety notes at the end."""

            user_prompt = f"""Based on the following SOP documents and training materials, provide a step-by-step procedure for: {query}

Context from documents:
{context}

Please provide a clear, numbered procedure with safety notes. Include specific details from the documents."""

            if model_type == "openai":
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                answer = response.content
            elif model_type == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(full_prompt)
                answer = response.text
            elif model_type == "openrouter":
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.openrouter_model.invoke(messages)
                answer = response.content
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Add citations
            answer += "\n\n**Sources:**\n"
            for source in sources:
                answer += f"- {source['document_title']} (v{source['document_version']})\n"
            
            return answer
            
        except Exception as e:
            print(f"Error generating AI procedure response with {model_type}: {e}")
            return self._generate_procedure_response(query, context, sources)
    
    def _generate_ai_regular_response(self, query: str, context: str, sources: List[Dict], model_type: str = "openai") -> str:
        """Generate a regular response using AI."""
        try:
            system_prompt = """You are an expert Food & Beverage training assistant. Your role is to provide helpful, accurate information based on SOP documents and training materials.

When answering questions:
1. Use information from the provided context
2. Be helpful and informative
3. Include relevant details from the documents
4. Always cite your sources
5. If the information isn't in the context, say so clearly"""

            user_prompt = f"""Based on the following SOP documents and training materials, answer this question: {query}

Context from documents:
{context}

Please provide a helpful answer using information from the documents."""

            if model_type == "openai":
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                answer = response.content
            elif model_type == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(full_prompt)
                answer = response.text
            elif model_type == "openrouter":
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.openrouter_model.invoke(messages)
                answer = response.content
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Add citations
            answer += "\n\n**Sources:**\n"
            for source in sources:
                answer += f"- {source['document_title']} (v{source['document_version']})\n"
            
            return answer
            
        except Exception as e:
            print(f"Error generating AI regular response with {model_type}: {e}")
            return self._generate_regular_response(query, context, sources)
    
    def _generate_procedure_response(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate a step-by-step procedure response (fallback)."""
        # This is a simplified response generator
        # In production, you'd use a proper LLM
        
        response = f"Based on the SOPs and training documents, here's the step-by-step procedure for: {query}\n\n"
        
        # Extract potential steps from context
        lines = context.split('\n')
        steps = []
        step_num = 1
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'procedure', 'process', 'method', 'instruction']):
                if len(line.strip()) > 20:  # Only include substantial lines
                    steps.append(f"{step_num}. {line.strip()}")
                    step_num += 1
        
        if steps:
            response += "\n".join(steps[:10])  # Limit to 10 steps
        else:
            response += "1. Review the relevant SOP document\n"
            response += "2. Follow the documented procedures\n"
            response += "3. Ensure all safety protocols are followed\n"
            response += "4. Complete the required documentation\n"
        
        # Add safety note
        response += "\n\n⚠️ **Safety Note**: Always follow proper safety protocols and consult with your manager if you're unsure about any step."
        
        # Add citations
        response += "\n\n**Sources:**\n"
        for source in sources:
            response += f"- {source['document_title']} (v{source['document_version']})\n"
        
        return response
    
    def _generate_regular_response(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate a regular response."""
        # This is a simplified response generator
        # In production, you'd use a proper LLM
        
        response = f"Based on the SOPs and training documents, here's what I found regarding: {query}\n\n"
        
        # Extract relevant information from context
        relevant_info = []
        lines = context.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in query.lower().split()):
                if len(line.strip()) > 20:  # Only include substantial lines
                    relevant_info.append(line.strip())
        
        if relevant_info:
            response += "\n".join(relevant_info[:5])  # Limit to 5 relevant pieces
        else:
            response += "The documents contain relevant information about this topic. Please refer to the specific SOP documents for detailed procedures."
        
        # Add note about consulting documents
        response += "\n\n📋 **Note**: For complete and up-to-date procedures, always refer to the official SOP documents."
        
        # Add citations
        response += "\n\n**Sources:**\n"
        for source in sources:
            response += f"- {source['document_title']} (v{source['document_version']})\n"
        
        return response
    
    def get_ai_model_status(self) -> str:
        """Get the current AI model being used."""
        if self.llm:
            return "OpenAI GPT-3.5-turbo"
        elif self.gemini_model:
            return "Google Gemini Pro"
        elif self.openrouter_model:
            return "OpenRouter (GPT-3.5-turbo)"
        else:
            return "Fallback (No AI model available)"
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database status for debugging."""
        try:
            doc_count = self.db.query(Document).count()
            chunk_count = self.db.query(Chunk).count()
            
            # Get sample documents
            docs = self.db.query(Document).limit(5).all()
            doc_info = []
            for doc in docs:
                chunk_count_for_doc = self.db.query(Chunk).filter(
                    Chunk.chunk_metadata.op('->>')('document_id') == str(doc.id)
                ).count()
                doc_info.append({
                    "id": doc.id,
                    "title": doc.title,
                    "chunks": chunk_count_for_doc
                })
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "sample_documents": doc_info
            }
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close the database session."""
        self.db.close()
