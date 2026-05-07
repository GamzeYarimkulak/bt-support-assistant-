"""
Dynamic context weighting for hybrid retrieval.
Analyzes query characteristics and adjusts BM25/embedding weights accordingly.
"""

from typing import Dict, Any
import re
import structlog

logger = structlog.get_logger()


class DynamicWeightComputer:
    """
    Computes dynamic alpha weights for hybrid retrieval based on query characteristics.
    
    Strategy:
    - Short, technical queries → Lower alpha (favor embeddings for semantic matching)
    - Long, free-form queries → Higher alpha (favor BM25 for keyword matching)
    - Mixed queries → Balanced alpha
    """
    
    # Technical terms that indicate semantic search should be favored
    TECHNICAL_TERMS = {
        # Turkish technical terms
        "vpn", "outlook", "email", "şifre", "parola", "yazıcı", "printer",
        "ağ", "network", "bağlantı", "connection", "sürücü", "driver",
        "güncelleme", "update", "yükleme", "installation", "kurulum",
        "hata", "error", "sorun", "problem", "çözüm", "solution",
        "kimlik", "authentication", "doğrulama", "verification",
        "erişim", "access", "izin", "permission", "yetki", "authorization",
        # English technical terms
        "password", "reset", "login", "account", "server", "client",
        "database", "backup", "restore", "firewall", "security",
        "ssl", "tls", "certificate", "domain", "dns", "ip", "dhcp"
    }
    
    # Stop words (common words that don't add semantic value)
    STOP_WORDS = {
        "nasıl", "ne", "neden", "nerede", "ne zaman", "kim", "hangi",
        "how", "what", "why", "where", "when", "who", "which",
        "bir", "bu", "şu", "o", "ile", "için", "gibi", "kadar",
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "ve", "veya", "ya", "da", "de", "ki", "mi", "mı", "mu", "mü",
        "and", "or", "but", "with", "for", "to", "of", "in", "on", "at"
    }
    
    def __init__(
        self,
        min_query_length: int = 3,
        max_query_length: int = 50,
        technical_weight_penalty: float = 0.2,
        length_weight_factor: float = 0.1
    ):
        """
        Initialize dynamic weight computer.
        
        Args:
            min_query_length: Minimum query length to consider
            max_query_length: Maximum query length to consider
            technical_weight_penalty: How much to reduce alpha for technical queries
            length_weight_factor: How much query length affects alpha
        """
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.technical_weight_penalty = technical_weight_penalty
        self.length_weight_factor = length_weight_factor
        
        logger.info("dynamic_weight_computer_initialized",
                   min_length=min_query_length,
                   max_length=max_query_length)
    
    def compute_alpha(self, query: str) -> float:
        """
        Compute dynamic alpha weight based on query characteristics.
        
        Args:
            query: User query string
            
        Returns:
            Alpha value in [0.0, 1.0]
            - Lower alpha (0.2-0.4): Favor embeddings (semantic search)
            - Medium alpha (0.4-0.6): Balanced
            - Higher alpha (0.6-0.8): Favor BM25 (keyword search)
        """
        if not query or len(query.strip()) < self.min_query_length:
            # Very short queries → default balanced
            return 0.5
        
        query_lower = query.lower().strip()
        query_words = self._tokenize(query_lower)
        
        if not query_words:
            return 0.5
        
        # Analyze query characteristics
        query_length = len(query_words)
        technical_term_count = self._count_technical_terms(query_words)
        technical_ratio = technical_term_count / len(query_words) if query_words else 0
        
        # Base alpha (start with balanced)
        alpha = 0.5
        
        # Adjust based on query length
        # Shorter queries → favor embeddings (lower alpha)
        # Longer queries → favor BM25 (higher alpha)
        if query_length <= 3:
            # Very short queries (e.g., "VPN bağlantı")
            alpha = 0.3  # Strong semantic preference
        elif query_length <= 5:
            # Short queries (e.g., "Outlook şifre sıfırlama")
            alpha = 0.4  # Semantic preference
        elif query_length <= 10:
            # Medium queries → balanced
            alpha = 0.5
        elif query_length <= 15:
            # Longer queries → keyword preference
            alpha = 0.6
        else:
            # Very long queries → strong keyword preference
            alpha = 0.7
        
        # Adjust based on technical term ratio
        if technical_ratio > 0.3:
            # High technical content → favor semantic search
            alpha -= self.technical_weight_penalty
            logger.debug("technical_query_detected",
                        technical_ratio=technical_ratio,
                        adjusted_alpha=alpha)
        
        # Clamp to valid range
        alpha = max(0.2, min(0.8, alpha))
        
        logger.debug("dynamic_alpha_computed",
                    query_length=query_length,
                    technical_terms=technical_term_count,
                    technical_ratio=technical_ratio,
                    final_alpha=alpha)
        
        return alpha
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words, removing stop words.
        
        Args:
            text: Input text
            
        Returns:
            List of meaningful words
        """
        # Simple tokenization: split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        meaningful_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 1]
        
        return meaningful_words
    
    def _count_technical_terms(self, words: list[str]) -> int:
        """
        Count technical terms in the word list.
        
        Args:
            words: List of words
            
        Returns:
            Number of technical terms found
        """
        count = 0
        for word in words:
            if word in self.TECHNICAL_TERMS:
                count += 1
            # Also check for partial matches (e.g., "vpn" in "vpn'ye")
            for term in self.TECHNICAL_TERMS:
                if term in word or word in term:
                    count += 0.5  # Partial match
                    break
        
        return int(count)
    
    def get_query_characteristics(self, query: str) -> Dict[str, Any]:
        """
        Get detailed characteristics of a query for analysis.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query characteristics
        """
        query_lower = query.lower().strip()
        words = self._tokenize(query_lower)
        
        return {
            "word_count": len(words),
            "char_count": len(query),
            "technical_term_count": self._count_technical_terms(words),
            "technical_ratio": self._count_technical_terms(words) / len(words) if words else 0,
            "is_short": len(words) <= 3,
            "is_long": len(words) > 15,
            "computed_alpha": self.compute_alpha(query)
        }


















