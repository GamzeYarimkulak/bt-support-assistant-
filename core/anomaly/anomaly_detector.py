"""
Anomaly detection using statistical methods.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import structlog

logger = structlog.get_logger()


# ============================================================================
# PHASE 5: Anomaly Event Structures
# ============================================================================

@dataclass
class AnomalyThresholds:
    """
    Thresholds for classifying anomaly severity.
    
    Combined drift scores are compared against these thresholds:
    - Below warning: No anomaly (info)
    - warning <= score < critical: Warning level
    - score >= critical: Critical level
    """
    warning: float = 2.0
    critical: float = 3.5


@dataclass
class AnomalyEvent:
    """
    Detected anomaly event for a time window.
    
    Contains:
    - Time window boundaries
    - Severity level (info, warning, critical)
    - Anomaly score
    - Human-readable reasons
    """
    window_start: datetime
    window_end: datetime
    severity: str  # "info" | "warning" | "critical"
    score: float
    reasons: List[str]


class AnomalyDetector:
    """
    Detects anomalies in ticket features using multiple methods.
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        threshold: float = 2.5,
        contamination: float = 0.1
    ):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'statistical', 'combined')
            threshold: Z-score threshold for statistical method
            contamination: Expected proportion of anomalies (for isolation forest)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.model = None
        
        if method == "isolation_forest":
            self.model = IsolationForest(contamination=contamination, random_state=42)
        
        logger.info("anomaly_detector_initialized",
                   method=method,
                   threshold=threshold,
                   contamination=contamination)
    
    def fit(self, feature_vectors: np.ndarray):
        """
        Fit the anomaly detector on historical data.
        
        Args:
            feature_vectors: Array of feature vectors (n_samples x n_features)
        """
        if self.method == "isolation_forest" and self.model is not None:
            self.model.fit(feature_vectors)
            logger.info("anomaly_detector_fitted",
                       n_samples=len(feature_vectors),
                       n_features=feature_vectors.shape[1])
    
    def detect_statistical(
        self,
        feature_vectors: np.ndarray,
        baseline_mean: np.ndarray,
        baseline_std: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using statistical z-score method.
        
        Args:
            feature_vectors: Feature vectors to check
            baseline_mean: Mean of baseline distribution
            baseline_std: Standard deviation of baseline distribution
            
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        # Compute z-scores
        z_scores = np.abs((feature_vectors - baseline_mean) / (baseline_std + 1e-8))
        
        # Max z-score across features
        max_z_scores = np.max(z_scores, axis=1)
        
        # Flag anomalies
        is_anomaly = max_z_scores > self.threshold
        
        logger.debug("statistical_detection_completed",
                    n_samples=len(feature_vectors),
                    n_anomalies=np.sum(is_anomaly))
        
        return is_anomaly, max_z_scores
    
    def detect_isolation_forest(
        self,
        feature_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using isolation forest.
        
        Args:
            feature_vectors: Feature vectors to check
            
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Predict: -1 for anomalies, 1 for normal
        predictions = self.model.predict(feature_vectors)
        is_anomaly = predictions == -1
        
        # Get anomaly scores (more negative = more anomalous)
        scores = -self.model.score_samples(feature_vectors)
        
        logger.debug("isolation_forest_detection_completed",
                    n_samples=len(feature_vectors),
                    n_anomalies=np.sum(is_anomaly))
        
        return is_anomaly, scores
    
    def detect(
        self,
        feature_vectors: np.ndarray,
        baseline_mean: np.ndarray = None,
        baseline_std: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using configured method.
        
        Args:
            feature_vectors: Feature vectors to check
            baseline_mean: Mean for statistical method (optional)
            baseline_std: Std for statistical method (optional)
            
        Returns:
            List of anomaly results
        """
        if self.method == "statistical":
            if baseline_mean is None or baseline_std is None:
                raise ValueError("Statistical method requires baseline_mean and baseline_std")
            is_anomaly, scores = self.detect_statistical(feature_vectors, baseline_mean, baseline_std)
        
        elif self.method == "isolation_forest":
            is_anomaly, scores = self.detect_isolation_forest(feature_vectors)
        
        elif self.method == "combined":
            # Combine both methods
            iso_anomaly, iso_scores = self.detect_isolation_forest(feature_vectors)
            
            if baseline_mean is not None and baseline_std is not None:
                stat_anomaly, stat_scores = self.detect_statistical(
                    feature_vectors, baseline_mean, baseline_std
                )
                # Consensus: anomaly if both methods agree OR very high score in one
                is_anomaly = (iso_anomaly & stat_anomaly) | (iso_scores > 0.8) | (stat_scores > self.threshold * 1.5)
                scores = (iso_scores + stat_scores) / 2
            else:
                is_anomaly, scores = iso_anomaly, iso_scores
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Build results
        results = []
        for i, (anomaly, score) in enumerate(zip(is_anomaly, scores)):
            results.append({
                "index": i,
                "is_anomalous": bool(anomaly),
                "anomaly_score": float(score),
                "method": self.method
            })
        
        n_anomalies = sum(r["is_anomalous"] for r in results)
        logger.info("anomaly_detection_completed",
                   n_samples=len(feature_vectors),
                   n_anomalies=n_anomalies,
                   method=self.method)
        
        return results
    
    def compute_baseline_stats(
        self,
        feature_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute baseline statistics for statistical detection.
        
        Args:
            feature_vectors: Historical feature vectors
            
        Returns:
            Tuple of (mean, std)
        """
        mean = np.mean(feature_vectors, axis=0)
        std = np.std(feature_vectors, axis=0)
        
        logger.debug("baseline_stats_computed",
                    n_samples=len(feature_vectors),
                    n_features=len(mean))
        
        return mean, std


# ============================================================================
# PHASE 5: Threshold-Based Anomaly Detector
# ============================================================================

class ThresholdAnomalyDetector:
    """
    Simple threshold-based anomaly detector for drift scores (PHASE 5).
    
    Classifies windows as info/warning/critical based on combined drift scores
    and generates human-readable reasons.
    """
    
    def __init__(self, thresholds: AnomalyThresholds | None = None):
        """
        Initialize detector with thresholds.
        
        Args:
            thresholds: Anomaly severity thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or AnomalyThresholds()
        
        logger.info("threshold_anomaly_detector_initialized",
                   warning=self.thresholds.warning,
                   critical=self.thresholds.critical)
    
    def detect(self, drift_scores: List['DriftScore']) -> List[AnomalyEvent]:
        """
        Detect anomalies from drift scores.
        
        For each drift score:
        - Determines severity level based on thresholds
        - Generates human-readable reasons
        - Creates AnomalyEvent
        
        Args:
            drift_scores: List of DriftScore objects
            
        Returns:
            List of AnomalyEvent objects sorted by window_start
            
        Example:
            >>> detector = ThresholdAnomalyDetector()
            >>> events = detector.detect(drift_scores)
            >>> for event in events:
            ...     if event.severity != "info":
            ...         print(f"{event.window_start}: {event.severity} - {event.reasons}")
        """
        if not drift_scores:
            logger.warning("no_drift_scores_to_detect")
            return []
        
        logger.info("detecting_anomalies_from_drift",
                   num_scores=len(drift_scores))
        
        events = []
        
        for score in drift_scores:
            # Determine severity based on combined_score
            # Use raw combined_score (not normalized)
            severity, reasons = self._classify_anomaly(score)
            
            event = AnomalyEvent(
                window_start=score.window_start,
                window_end=score.window_end,
                severity=severity,
                score=score.combined_score,
                reasons=reasons
            )
            
            events.append(event)
        
        # Sort by window_start
        events.sort(key=lambda x: x.window_start)
        
        # Count by severity
        severity_counts = {}
        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        logger.info("anomaly_detection_completed",
                   total_windows=len(events),
                   severity_distribution=severity_counts)
        
        return events
    
    def _classify_anomaly(self, score: 'DriftScore') -> Tuple[str, List[str]]:
        """
        Classify anomaly severity and generate reasons.
        
        Args:
            score: DriftScore to classify
            
        Returns:
            Tuple of (severity, reasons)
        """
        reasons = []
        
        # Check volume z-score
        if abs(score.volume_zscore) > 2.0:
            direction = "spike" if score.volume_zscore > 0 else "drop"
            reasons.append(
                f"Volume z-score {score.volume_zscore:.2f} ({direction} in total tickets)"
            )
        
        # Check category divergence
        if score.category_divergence > 0.2:
            reasons.append(
                f"Category divergence {score.category_divergence:.3f} (distribution shift detected)"
            )
        
        # Check embedding shift
        if score.embedding_shift > 0.15:
            reasons.append(
                f"Embedding shift {score.embedding_shift:.3f} (semantic drift detected)"
            )
        
        # Determine severity based on combined score
        # Note: combined_score is normalized to [0, 1] range
        if score.combined_score >= self.thresholds.critical:
            severity = "critical"
        elif score.combined_score >= self.thresholds.warning:
            severity = "warning"
        else:
            severity = "info"
            if not reasons:
                reasons.append("No significant anomalies detected")
        
        return severity, reasons

