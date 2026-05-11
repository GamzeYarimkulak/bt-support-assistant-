"""
Prompt templates and builders for RAG system.
"""

from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()


class PromptBuilder:
    """
    Builds prompts for the RAG system with strict "no source, no answer" policy.
    """
    
    SYSTEM_PROMPT = """You are a helpful IT support assistant for an enterprise ITSM system.

CRITICAL RULES:
1. You MUST ONLY answer based on the provided context documents
2. If the context does not contain enough information to answer the question, you MUST say: "I don't have enough information in the knowledge base to answer this question reliably."
3. NEVER make up information or use knowledge outside the provided context
4. Always cite which document(s) you used to formulate your answer
5. Be concise and professional

When answering:
- Reference specific tickets or documents by their ID
- If multiple sources contain relevant information, synthesize them
- If sources contradict each other, mention the contradiction
- Maintain user privacy by not exposing sensitive information"""
    
    USER_PROMPT_TEMPLATE = """Context Documents:
{context}

---

User Question: {query}

Instructions: Answer the question using ONLY the information from the context documents above. If you cannot answer based on the provided context, explicitly state that you don't have enough information."""
    
    def __init__(self):
        """Initialize prompt builder."""
        logger.info("prompt_builder_initialized")
    
    def build_context_string(self, documents: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            max_length: Maximum character length for context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_id = doc.get("id", doc.get("doc_id", f"doc_{i}"))
            doc_type = doc.get("doc_type", "document")
            title = doc.get("title", "Untitled")
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            
            doc_str = f"[{i}] {doc_type.upper()} ID: {doc_id}\n"
            doc_str += f"Title: {title}\n"
            doc_str += f"Relevance Score: {score:.3f}\n"
            doc_str += f"Content: {text}\n"
            doc_str += "-" * 80 + "\n"
            
            if current_length + len(doc_str) > max_length:
                context_parts.append(f"\n... (truncated, showing top {i-1} documents)")
                break
            
            context_parts.append(doc_str)
            current_length += len(doc_str)
        
        return "\n".join(context_parts)
    
    def build_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_context_length: int = 2000
    ) -> Dict[str, str]:
        """
        Build complete prompt for the LLM.
        
        Args:
            query: User query
            documents: Retrieved documents
            max_context_length: Maximum length for context
            
        Returns:
            Dictionary with system and user prompts
        """
        context = self.build_context_string(documents, max_length=max_context_length)
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        
        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_prompt
        }
    
    def build_conversation_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Build prompt with conversation history.
        
        Args:
            query: Current user query
            documents: Retrieved documents
            conversation_history: List of previous messages
            max_context_length: Maximum length for context
            
        Returns:
            Dictionary with system prompt and message history
        """
        context = self.build_context_string(documents, max_length=max_context_length)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Add conversation history
        for msg in conversation_history[-5:]:  # Keep last 5 exchanges
            messages.append(msg)
        
        # Add current query with context
        current_user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        messages.append({"role": "user", "content": current_user_prompt})
        
        return {"messages": messages}
    
    def extract_sources_from_context(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents for citation.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of source metadata
        """
        sources = []
        for doc in documents:
            source = {
                "doc_id": doc.get("id", doc.get("doc_id", "unknown")),
                "doc_type": doc.get("doc_type", "document"),
                "title": doc.get("title", "Untitled"),
                "score": doc.get("score", 0.0)
            }
            sources.append(source)
        
        return sources


