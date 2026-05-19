"""
PHASE 5: Real Anomaly Detection Engine

This module implements time-series anomaly detection over IT support tickets.
It monitors ticket streams across time windows and detects:
- Volume spikes (statistical outliers in ticket count)
- Category distribution shifts (changes in issue types)
- Semantic drift (changes in content/meaning of tickets)

The engine computes a combined anomaly score and severity level for each
time window, enabling proactive monitoring and early warning systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================
# DATA MODELS
# ============================================

@dataclass
class AnomalyTicket:
    """
    Represents a single ticket for anomaly detection.
    
    Attributes:
        ticket_id: Unique identifier
        created_at: Timestamp when ticket was created
        category: Primary category (e.g., "Outlook", "VPN", "Printer")
        subcategory: Optional subcategory for finer granularity
        priority: Ticket priority (e.g., "Low", "Medium", "High", "Critical")
        embedding: Vector representation of ticket content (for semantic analysis)
    """
    ticket_id: str
    created_at: datetime
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class WindowStats:
    """
    Statistics and anomaly scores for a single time window.
    
    Attributes:
        window_start: Start of time window
        window_end: End of time window
        total_tickets: Number of tickets in this window
        volume_z: Z-score for volume anomaly (how unusual is the ticket count?)
        category_divergence: JS divergence measuring category distribution shift
        semantic_drift: Cosine distance measuring content drift
        combined_score: Combined anomaly score [0, 1]
        severity: Severity level ("normal", "info", "warning", "critical")
        reasons: Human-readable list of detected anomalies
    """
    window_start: datetime
    window_end: datetime
    total_tickets: int
    volume_z: Optional[float] = None
    category_divergence: Optional[float] = None
    semantic_drift: Optional[float] = None
    combined_score: float = 0.0
    severity: str = "normal"
    reasons: List[str] = field(default_factory=list)


@dataclass
class AnomalyEvent:
    """
    Represents a detected anomaly event (non-normal window).
    
    Attributes:
        window_start: Start of anomalous window
        window_end: End of anomalous window
        severity: Severity level ("info", "warning", "critical")
        score: Combined anomaly score
        reasons: List of specific anomalies detected
    """
    window_start: datetime
    window_end: datetime
    severity: str
    score: float
    reasons: List[str]


# ============================================
# WINDOWING
# ============================================

def build_time_windows(
    tickets: List[AnomalyTicket],
    window_size: timedelta = timedelta(days=1),
) -> Tuple[List[List[AnomalyTicket]], List[Tuple[datetime, datetime]]]:
    """
    Partition tickets into contiguous time windows.
    
    Args:
        tickets: List of tickets to partition
        window_size: Duration of each window (default: 1 day)
    
    Returns:
        Tuple of:
        - List of ticket lists (one per window)
        - List of (window_start, window_end) tuples
    
    Example:
        If window_size = 1 day and tickets span 2024-12-01 to 2024-12-03:
        - Window 0: tickets on 2024-12-01
        - Window 1: tickets on 2024-12-02
        - Window 2: tickets on 2024-12-03
    """
    if not tickets:
        return [], []
    
    # Sort tickets by creation time
    sorted_tickets = sorted(tickets, key=lambda t: t.created_at)
    
    # Find time range
    min_time = sorted_tickets[0].created_at
    max_time = sorted_tickets[-1].created_at
    
    # Build window boundaries
    windows = []
    window_bounds = []
    
    current_start = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_start <= max_time:
        current_end = current_start + window_size
        
        # Get tickets in this window
        window_tickets = [
            t for t in sorted_tickets
            if current_start <= t.created_at < current_end
        ]
        
        windows.append(window_tickets)
        window_bounds.append((current_start, current_end))
        
        current_start = current_end
    
    return windows, window_bounds


# ============================================
# STATISTICAL HELPERS
# ============================================

def compute_volume_zscore(
    current_count: int,
    baseline_counts: List[int],
) -> Optional[float]:
    """
    Compute z-score for volume anomaly detection.
    
    Args:
        current_count: Ticket count in current window
        baseline_counts: Ticket counts in previous windows
    
    Returns:
        Z-score (standard deviations from mean), or None if insufficient data
    
    Interpretation:
        |z| > 3: Very unusual (likely anomaly)
        |z| > 2: Unusual
        |z| > 1.5: Slightly unusual
        |z| <= 1.5: Normal
    """
    if not baseline_counts:
        return None
    
    mean = np.mean(baseline_counts)
    std = np.std(baseline_counts)
    
    if std == 0:
        # No variation in baseline
        if current_count == mean:
            return 0.0
        else:
            # Any deviation is infinite z-score, cap at 5
            return 5.0 if current_count > mean else -5.0
    
    z_score = (current_count - mean) / std
    return float(z_score)


def compute_jensen_shannon_divergence(
    dist1: Dict[str, float],
    dist2: Dict[str, float],
) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    JS divergence is a symmetrized and smoothed version of KL divergence.
    It measures how different two distributions are.
    
    Args:
        dist1: First distribution (dict of category -> probability)
        dist2: Second distribution
    
    Returns:
        JS divergence in [0, 1] (0 = identical, 1 = completely different)
    
    Note:
        JS divergence is always finite and symmetric, unlike KL divergence.
    """
    # Get all categories
    all_categories = set(dist1.keys()) | set(dist2.keys())
    
    if not all_categories:
        return 0.0
    
    # Convert to probability arrays
    p = np.array([dist1.get(cat, 0.0) for cat in all_categories])
    q = np.array([dist2.get(cat, 0.0) for cat in all_categories])
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute average distribution
    m = (p + q) / 2.0
    
    # Compute KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # JS divergence is average of two KL divergences
    js = (kl_pm + kl_qm) / 2.0
    
    # Normalize to [0, 1] (theoretical max is log(2) ≈ 0.693)
    js_normalized = js / np.log(2)
    
    return float(np.clip(js_normalized, 0.0, 1.0))


def compute_semantic_drift(
    current_embeddings: List[np.ndarray],
    baseline_embeddings: List[np.ndarray],
) -> Optional[float]:
    """
    Compute semantic drift using cosine distance between mean embeddings.
    
    Args:
        current_embeddings: Embeddings from current window
        baseline_embeddings: Embeddings from baseline windows
    
    Returns:
        Cosine distance in [0, 1], or None if insufficient data
        (0 = identical, 1 = orthogonal/completely different)
    
    Interpretation:
        distance > 0.3: Significant drift
        distance > 0.2: Moderate drift
        distance > 0.1: Slight drift
        distance <= 0.1: Normal variation
    """
    if not current_embeddings or not baseline_embeddings:
        return None
    
    # Compute mean embeddings
    current_mean = np.mean(current_embeddings, axis=0)
    baseline_mean = np.mean(baseline_embeddings, axis=0)
    
    # Compute cosine similarity
    dot_product = np.dot(current_mean, baseline_mean)
    norm_product = np.linalg.norm(current_mean) * np.linalg.norm(baseline_mean)
    
    if norm_product == 0:
        return None
    
    cosine_sim = dot_product / norm_product
    
    # Convert to distance (0 = same, 1 = orthogonal)
    cosine_dist = 1.0 - cosine_sim
    
    return float(np.clip(cosine_dist, 0.0, 1.0))


# ============================================
# FEATURE COMPUTATION
# ============================================

def compute_window_stats(
    windows: List[List[AnomalyTicket]],
    window_bounds: List[Tuple[datetime, datetime]],
    min_baseline_windows: int = 3,
) -> List[WindowStats]:
    """
    Compute anomaly statistics for each time window.
    
    For each window, computes:
    - Volume z-score (compared to historical baseline)
    - Category divergence (distribution shift)
    - Semantic drift (content change)
    
    Args:
        windows: List of ticket lists (one per window)
        window_bounds: List of (start, end) tuples for each window
        min_baseline_windows: Minimum number of previous windows needed for baseline
    
    Returns:
        List of WindowStats objects
    
    Note:
        The first few windows may have partial stats due to insufficient baseline.
    """
    stats_list = []
    
    for i, (tickets, (start, end)) in enumerate(zip(windows, window_bounds)):
        current_count = len(tickets)
        
        # Initialize stats
        volume_z = None
        category_div = None
        semantic_drift_val = None
        
        # Get baseline windows (all previous windows)
        baseline_windows = windows[:i]
        
        if len(baseline_windows) >= min_baseline_windows:
            # ========================================
            # 1. VOLUME Z-SCORE
            # ========================================
            baseline_counts = [len(w) for w in baseline_windows]
            volume_z = compute_volume_zscore(current_count, baseline_counts)
            
            # ========================================
            # 2. CATEGORY DIVERGENCE
            # ========================================
            # Current window category distribution
            current_categories = [t.category for t in tickets if t.category]
            if current_categories:
                current_counter = Counter(current_categories)
                current_total = sum(current_counter.values())
                current_dist = {
                    cat: count / current_total
                    for cat, count in current_counter.items()
                }
                
                # Baseline category distribution (all previous windows)
                baseline_categories = [
                    t.category
                    for w in baseline_windows
                    for t in w
                    if t.category
                ]
                
                if baseline_categories:
                    baseline_counter = Counter(baseline_categories)
                    baseline_total = sum(baseline_counter.values())
                    baseline_dist = {
                        cat: count / baseline_total
                        for cat, count in baseline_counter.items()
                    }
                    
                    # Compute JS divergence
                    category_div = compute_jensen_shannon_divergence(
                        current_dist, baseline_dist
                    )
            
            # ========================================
            # 3. SEMANTIC DRIFT
            # ========================================
            current_embeddings = [t.embedding for t in tickets if t.embedding is not None]
            baseline_embeddings = [
                t.embedding
                for w in baseline_windows
                for t in w
                if t.embedding is not None
            ]
            
            if current_embeddings and baseline_embeddings:
                semantic_drift_val = compute_semantic_drift(
                    current_embeddings, baseline_embeddings
                )
        
        # Create WindowStats
        stats = WindowStats(
            window_start=start,
            window_end=end,
            total_tickets=current_count,
            volume_z=volume_z,
            category_divergence=category_div,
            semantic_drift=semantic_drift_val,
        )
        
        stats_list.append(stats)
    
    return stats_list


# ============================================
# COMBINED SCORE & SEVERITY
# ============================================

def combine_scores(
    volume_z: Optional[float],
    category_divergence: Optional[float],
    semantic_drift: Optional[float],
    w_volume: float = 0.3,
    w_category: float = 0.3,
    w_semantic: float = 0.4,
) -> float:
    """
    Combine individual anomaly scores into a single score.
    
    Args:
        volume_z: Z-score for volume anomaly
        category_divergence: JS divergence for category shift
        semantic_drift: Cosine distance for semantic drift
        w_volume: Weight for volume component
        w_category: Weight for category component
        w_semantic: Weight for semantic component
    
    Returns:
        Combined score in [0, 1]
        (0 = normal, 1 = highly anomalous)
    
    Normalization:
        - Volume: |z| <= 3 maps to [0, 1], clamped above
        - Category: divergence [0, 1] (already normalized)
        - Semantic: drift [0, 0.5] maps to [0, 1], clamped above
    """
    scores = []
    weights = []
    
    # 1. Volume component
    if volume_z is not None:
        # Normalize |z| to [0, 1], clamp at 3
        volume_score = min(abs(volume_z) / 3.0, 1.0)
        scores.append(volume_score)
        weights.append(w_volume)
    
    # 2. Category component
    if category_divergence is not None:
        # Already in [0, 1]
        scores.append(category_divergence)
        weights.append(w_category)
    
    # 3. Semantic component
    if semantic_drift is not None:
        # Normalize [0, 0.5] to [0, 1], clamp at 0.5
        semantic_score = min(semantic_drift / 0.5, 1.0)
        scores.append(semantic_score)
        weights.append(w_semantic)
    
    # Weighted average
    if not scores:
        return 0.0
    
    # Renormalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    normalized_weights = [w / total_weight for w in weights]
    
    combined = sum(s * w for s, w in zip(scores, normalized_weights))
    
    return float(np.clip(combined, 0.0, 1.0))


def determine_severity(
    combined_score: float,
    threshold_info: float = 0.3,
    threshold_warning: float = 0.6,
    threshold_critical: float = 0.8,
) -> str:
    """
    Determine severity level based on combined score.
    
    Args:
        combined_score: Combined anomaly score [0, 1]
        threshold_info: Threshold for "info" level
        threshold_warning: Threshold for "warning" level
        threshold_critical: Threshold for "critical" level
    
    Returns:
        Severity level: "normal", "info", "warning", or "critical"
    
    Interpretation:
        - normal: Business as usual
        - info: Slight anomaly, worth noting
        - warning: Moderate anomaly, investigate
        - critical: Severe anomaly, immediate action needed
    """
    if combined_score >= threshold_critical:
        return "critical"
    elif combined_score >= threshold_warning:
        return "warning"
    elif combined_score >= threshold_info:
        return "info"
    else:
        return "normal"


def generate_reasons(
    volume_z: Optional[float],
    category_divergence: Optional[float],
    semantic_drift: Optional[float],
    volume_threshold: float = 1.5,
    category_threshold: float = 0.3,
    semantic_threshold: float = 0.15,
) -> List[str]:
    """
    Generate human-readable reasons for detected anomalies.
    
    Args:
        volume_z: Volume z-score
        category_divergence: Category JS divergence
        semantic_drift: Semantic cosine distance
        volume_threshold: Z-score threshold for reporting
        category_threshold: Divergence threshold for reporting
        semantic_threshold: Drift threshold for reporting
    
    Returns:
        List of reason strings
    
    Example:
        ["Volume spike detected (z = 2.8)", "Category distribution shifted"]
    """
    reasons = []
    
    if volume_z is not None and abs(volume_z) > volume_threshold:
        direction = "spike" if volume_z > 0 else "drop"
        reasons.append(f"Volume {direction} detected (z = {volume_z:.2f})")
    
    if category_divergence is not None and category_divergence > category_threshold:
        reasons.append(
            f"Category distribution shifted (divergence = {category_divergence:.3f})"
        )
    
    if semantic_drift is not None and semantic_drift > semantic_threshold:
        reasons.append(
            f"Semantic drift detected (distance = {semantic_drift:.3f})"
        )
    
    return reasons


def finalize_window_stats(stats_list: List[WindowStats]) -> List[WindowStats]:
    """
    Finalize window stats by computing combined scores and severity levels.
    
    Args:
        stats_list: List of WindowStats with individual scores
    
    Returns:
        Same list with combined_score, severity, and reasons populated
    """
    for stats in stats_list:
        # Compute combined score
        stats.combined_score = combine_scores(
            stats.volume_z,
            stats.category_divergence,
            stats.semantic_drift,
        )
        
        # Determine severity
        stats.severity = determine_severity(stats.combined_score)
        
        # Generate reasons
        stats.reasons = generate_reasons(
            stats.volume_z,
            stats.category_divergence,
            stats.semantic_drift,
        )
    
    return stats_list


# ============================================
# HIGH-LEVEL API
# ============================================

def analyze_ticket_stream(
    tickets: List[AnomalyTicket],
    window_size: timedelta = timedelta(days=1),
    min_baseline_windows: int = 3,
) -> Tuple[List[WindowStats], List[AnomalyEvent]]:
    """
    Analyze a stream of tickets for anomalies.
    
    This is the main entry point for anomaly detection.
    
    Args:
        tickets: List of tickets to analyze
        window_size: Duration of each time window
        min_baseline_windows: Minimum baseline windows for statistical analysis
    
    Returns:
        Tuple of:
        - List of WindowStats for all windows
        - List of AnomalyEvent for anomalous windows (severity != "normal")
    
    Example:
        ```python
        tickets = [AnomalyTicket(...), ...]
        stats, events = analyze_ticket_stream(tickets)
        
        for event in events:
            print(f"Anomaly: {event.severity} at {event.window_start}")
            print(f"  Reasons: {', '.join(event.reasons)}")
        ```
    
    Note:
        - The first min_baseline_windows may have severity "normal" due to
          insufficient baseline data.
        - Embeddings can be None; semantic drift will be skipped if missing.
    """
    if not tickets:
        return [], []
    
    # Step 1: Build time windows
    windows, window_bounds = build_time_windows(tickets, window_size)
    
    # Step 2: Compute per-window statistics
    stats_list = compute_window_stats(windows, window_bounds, min_baseline_windows)
    
    # Step 3: Finalize (combined scores, severity, reasons)
    stats_list = finalize_window_stats(stats_list)
    
    # Step 4: Extract anomaly events (non-normal windows)
    events = []
    for stats in stats_list:
        if stats.severity != "normal":
            event = AnomalyEvent(
                window_start=stats.window_start,
                window_end=stats.window_end,
                severity=stats.severity,
                score=stats.combined_score,
                reasons=stats.reasons,
            )
            events.append(event)
    
    return stats_list, events

