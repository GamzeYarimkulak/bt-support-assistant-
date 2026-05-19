"""
Feature extraction from tickets for anomaly detection.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import structlog

logger = structlog.get_logger()


# ============================================================================
# PHASE 5: New Data Structures for Time-Window Anomaly Detection
# ============================================================================

@dataclass
class TicketFeatures:
    """
    Features extracted from a single ITSM ticket.
    
    This structure contains per-ticket information used for anomaly detection:
    - Identifiers and timestamps
    - Categorical metadata
    - Semantic embedding vector
    """
    ticket_id: str
    created_at: datetime
    category: str | None
    priority: str | None
    embedding: np.ndarray
    
    def __post_init__(self):
        """Ensure embedding is a numpy array."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


@dataclass
class WindowStats:
    """
    Aggregated statistics for a time window of tickets.
    
    Contains:
    - Time window boundaries
    - Ticket counts
    - Category and priority distributions
    - Centroid embedding (mean of all embeddings in window)
    """
    window_start: datetime
    window_end: datetime
    total_tickets: int
    counts_by_category: Dict[str, int]
    counts_by_priority: Dict[str, int]
    centroid_embedding: np.ndarray
    
    def __post_init__(self):
        """Ensure centroid_embedding is a numpy array."""
        if not isinstance(self.centroid_embedding, np.ndarray):
            self.centroid_embedding = np.array(self.centroid_embedding)


class FeatureExtractor:
    """
    Extracts features from ITSM tickets for anomaly detection.
    Features include: ticket counts, embedding aggregates, category distributions.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize feature extractor.
        
        Args:
            embedding_dim: Dimension of ticket embeddings
        """
        self.embedding_dim = embedding_dim
        logger.info("feature_extractor_initialized", embedding_dim=embedding_dim)
    
    def extract_temporal_features(
        self,
        tickets: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Extract temporal features from tickets.
        
        Args:
            tickets: List of ticket dictionaries
            window_hours: Time window in hours
            
        Returns:
            Dictionary of temporal features
        """
        if not tickets:
            return self._empty_temporal_features()
        
        # Count statistics
        total_count = len(tickets)
        
        # Priority distribution
        priorities = [t.get("priority", "unknown") for t in tickets]
        priority_dist = self._compute_distribution(priorities)
        
        # Category distribution
        categories = [t.get("category", "unknown") for t in tickets]
        category_dist = self._compute_distribution(categories)
        
        # Status distribution
        statuses = [t.get("status", "unknown") for t in tickets]
        status_dist = self._compute_distribution(statuses)
        
        features = {
            "total_count": total_count,
            "count_per_hour": total_count / window_hours,
            "priority_distribution": priority_dist,
            "category_distribution": category_dist,
            "status_distribution": status_dist,
            "window_hours": window_hours
        }
        
        logger.debug("temporal_features_extracted",
                    total_count=total_count,
                    window_hours=window_hours)
        
        return features
    
    def extract_semantic_features(
        self,
        tickets: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract semantic features from ticket embeddings.
        
        Args:
            tickets: List of ticket dictionaries
            embeddings: Ticket embeddings (n_tickets x embedding_dim)
            
        Returns:
            Dictionary of semantic features
        """
        if not tickets or embeddings is None or len(embeddings) == 0:
            return self._empty_semantic_features()
        
        # Embedding statistics
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        # Pairwise similarity statistics
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        avg_similarity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        std_similarity = np.std(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        features = {
            "mean_embedding": mean_embedding.tolist(),
            "std_embedding": std_embedding.tolist(),
            "avg_pairwise_similarity": float(avg_similarity),
            "std_pairwise_similarity": float(std_similarity),
            "embedding_centroid_norm": float(np.linalg.norm(mean_embedding))
        }
        
        logger.debug("semantic_features_extracted",
                    num_tickets=len(tickets),
                    avg_similarity=avg_similarity)
        
        return features
    
    def extract_combined_features(
        self,
        tickets: List[Dict[str, Any]],
        embeddings: np.ndarray,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Extract both temporal and semantic features.
        
        Args:
            tickets: List of ticket dictionaries
            embeddings: Ticket embeddings
            window_hours: Time window in hours
            
        Returns:
            Combined feature dictionary
        """
        temporal = self.extract_temporal_features(tickets, window_hours)
        semantic = self.extract_semantic_features(tickets, embeddings)
        
        return {
            "temporal": temporal,
            "semantic": semantic,
            "timestamp": datetime.now().isoformat()
        }
    
    def _compute_distribution(self, values: List[str]) -> Dict[str, float]:
        """
        Compute distribution of categorical values.
        
        Args:
            values: List of categorical values
            
        Returns:
            Dictionary mapping value to proportion
        """
        if not values:
            return {}
        
        total = len(values)
        counts = {}
        
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        
        distribution = {
            key: count / total
            for key, count in counts.items()
        }
        
        return distribution
    
    def _empty_temporal_features(self) -> Dict[str, Any]:
        """Return empty temporal features."""
        return {
            "total_count": 0,
            "count_per_hour": 0.0,
            "priority_distribution": {},
            "category_distribution": {},
            "status_distribution": {},
            "window_hours": 0
        }
    
    def _empty_semantic_features(self) -> Dict[str, Any]:
        """Return empty semantic features."""
        return {
            "mean_embedding": [0.0] * self.embedding_dim,
            "std_embedding": [0.0] * self.embedding_dim,
            "avg_pairwise_similarity": 0.0,
            "std_pairwise_similarity": 0.0,
            "embedding_centroid_norm": 0.0
        }


# ============================================================================
# PHASE 5: Feature Extraction Functions
# ============================================================================

def extract_ticket_features(
    tickets: List['ITSMTicket'],
    embedder: Any  # EmbeddingRetriever or similar
) -> List[TicketFeatures]:
    """
    Extract features from ITSM tickets for anomaly detection (PHASE 5).
    
    This function converts ITSMTicket objects into TicketFeatures by:
    1. Building a combined text representation
    2. Computing embeddings using the same model as retrieval
    3. Extracting metadata (category, priority, timestamp)
    
    Args:
        tickets: List of ITSMTicket objects
        embedder: Embedding encoder with encode() method (e.g., EmbeddingRetriever)
        
    Returns:
        List of TicketFeatures
        
    Example:
        >>> from data_pipeline.ingestion import load_itsm_tickets_from_csv
        >>> from core.retrieval.embedding_retriever import EmbeddingRetriever
        >>> tickets = load_itsm_tickets_from_csv("data/tickets.csv")
        >>> embedder = EmbeddingRetriever()
        >>> embedder.load_model()
        >>> features = extract_ticket_features(tickets, embedder)
    """
    if not tickets:
        logger.warning("no_tickets_to_extract_features")
        return []
    
    logger.info("extracting_ticket_features", num_tickets=len(tickets))
    
    # Build text representations
    texts = []
    for ticket in tickets:
        text_parts = []
        if ticket.short_description:
            text_parts.append(ticket.short_description)
        if ticket.description:
            text_parts.append(ticket.description)
        if ticket.resolution:
            text_parts.append(ticket.resolution)
        
        combined_text = " ".join(text_parts) if text_parts else "empty"
        texts.append(combined_text)
    
    # Compute embeddings using the embedder
    try:
        embeddings = embedder.encode(texts, normalize=True)
    except Exception as e:
        logger.error("embedding_computation_failed", error=str(e))
        raise
    
    # Create TicketFeatures objects
    features = []
    for i, ticket in enumerate(tickets):
        feature = TicketFeatures(
            ticket_id=ticket.ticket_id,
            created_at=ticket.created_at,
            category=ticket.category if ticket.category else None,
            priority=ticket.priority if ticket.priority else None,
            embedding=embeddings[i]
        )
        features.append(feature)
    
    logger.info("ticket_features_extracted", 
               num_features=len(features),
               embedding_dim=embeddings.shape[1])
    
    return features


def aggregate_time_windows(
    features: List[TicketFeatures],
    window: str = "1D"
) -> List[WindowStats]:
    """
    Aggregate ticket features into time windows (PHASE 5).
    
    Groups tickets by time windows and computes aggregated statistics:
    - Total ticket counts
    - Category and priority histograms
    - Centroid embedding (mean of all embeddings)
    
    Args:
        features: List of TicketFeatures
        window: Pandas-compatible window size (e.g., "1D", "7D", "1H")
        
    Returns:
        List of WindowStats sorted by window_start
        
    Example:
        >>> window_stats = aggregate_time_windows(features, window="1D")
        >>> for ws in window_stats:
        ...     print(f"{ws.window_start}: {ws.total_tickets} tickets")
    """
    if not features:
        logger.warning("no_features_to_aggregate")
        return []
    
    logger.info("aggregating_time_windows", 
               num_features=len(features),
               window=window)
    
    # Create DataFrame for easier grouping
    data = {
        'ticket_id': [f.ticket_id for f in features],
        'created_at': [f.created_at for f in features],
        'category': [f.category if f.category else 'unknown' for f in features],
        'priority': [f.priority if f.priority else 'unknown' for f in features],
        'embedding_idx': list(range(len(features)))
    }
    
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')
    
    # Group by time windows
    df['window'] = df['created_at'].dt.floor(window)
    
    window_stats_list = []
    
    for window_start, group in df.groupby('window'):
        window_end = window_start + pd.Timedelta(window)
        
        # Get indices for this window
        indices = group['embedding_idx'].tolist()
        
        # Compute category counts
        category_counts = group['category'].value_counts().to_dict()
        
        # Compute priority counts
        priority_counts = group['priority'].value_counts().to_dict()
        
        # Compute centroid embedding
        window_embeddings = np.array([features[idx].embedding for idx in indices])
        centroid = np.mean(window_embeddings, axis=0)
        
        ws = WindowStats(
            window_start=window_start.to_pydatetime(),
            window_end=window_end.to_pydatetime(),
            total_tickets=len(group),
            counts_by_category=category_counts,
            counts_by_priority=priority_counts,
            centroid_embedding=centroid
        )
        
        window_stats_list.append(ws)
    
    # Sort by window_start
    window_stats_list.sort(key=lambda x: x.window_start)
    
    logger.info("time_windows_aggregated",
               num_windows=len(window_stats_list),
               window_size=window)
    
    return window_stats_list


