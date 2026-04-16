"""
BM25-based lexical retrieval for document search.
"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import structlog
import numpy as np

logger = structlog.get_logger()


class BM25Retriever:
    """
    BM25 retriever for lexical search over documents.
    Provides fast keyword-based retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        
        logger.info("bm25_retriever_initialized", k1=k1, b=b)
    
    def index_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to index
        """
        self.documents = documents
        
        # Tokenize all documents
        self.tokenized_corpus = [
            doc[text_field].lower().split() 
            for doc in documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("bm25_index_built", num_documents=len(documents))
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if self.bm25 is None or not self.documents:
            logger.warning("bm25_search_attempted_without_index")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include docs with positive scores
                result = self.documents[idx].copy()
                result["score"] = float(scores[idx])
                result["retrieval_method"] = "bm25"
                results.append(result)
        
        logger.debug("bm25_search_completed", 
                    query=query, 
                    num_results=len(results))
        
        return results
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Search query
            
        Returns:
            Array of scores for all documents
        """
        if self.bm25 is None:
            return np.array([])
        
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed corpus.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.documents:
            return {"num_documents": 0, "indexed": False}
        
        avg_doc_length = np.mean([len(doc) for doc in self.tokenized_corpus])
        
        return {
            "num_documents": len(self.documents),
            "indexed": self.bm25 is not None,
            "avg_doc_length": float(avg_doc_length),
            "k1": self.k1,
            "b": self.b
        }


